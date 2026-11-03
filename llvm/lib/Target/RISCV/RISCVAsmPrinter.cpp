//===-- RISCVAsmPrinter.cpp - RISC-V LLVM assembly writer -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the RISC-V assembly language.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVInstPrinter.h"
#include "MCTargetDesc/RISCVMCAsmInfo.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "MCTargetDesc/RISCVTargetStreamer.h"
#include "RISCV.h"
#include "RISCVConstantPoolValue.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVRegisterInfo.h"
#include "TargetInfo/RISCVTargetInfo.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/RISCVISAInfo.h"
#include "llvm/Transforms/Instrumentation/HWAddressSanitizer.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

STATISTIC(RISCVNumInstrsCompressed,
          "Number of RISC-V Compressed instructions emitted");

namespace llvm {
extern const SubtargetFeatureKV RISCVFeatureKV[RISCV::NumSubtargetFeatures];
} // namespace llvm

namespace {
class RISCVAsmPrinter : public AsmPrinter {
public:
  static char ID;

private:
  const RISCVSubtarget *STI;
  // Guest-memory records waiting for the access they describe to be emitted.
  // A record may stay pending across a block boundary (see the adjacency scan)
  // while a later record is created, so this is a queue of (record, access)
  // pairs rather than a single pending pair.
  SmallVector<std::pair<const MachineInstr *, const MachineInstr *>, 4>
      PendingGuestStackMaps;

public:
  explicit RISCVAsmPrinter(TargetMachine &TM,
                           std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer), ID) {}

  StringRef getPassName() const override { return "RISC-V Assembly Printer"; }

  void LowerSTACKMAP(MCStreamer &OutStreamer, StackMaps &SM,
                     const MachineInstr &MI);

  void LowerPATCHPOINT(MCStreamer &OutStreamer, StackMaps &SM,
                       const MachineInstr &MI);

  void LowerSTATEPOINT(MCStreamer &OutStreamer, StackMaps &SM,
                       const MachineInstr &MI);

  bool runOnMachineFunction(MachineFunction &MF) override;

  void emitInstruction(const MachineInstr *MI) override;

  void emitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) override;

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &OS) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             const char *ExtraCode, raw_ostream &OS) override;

  // Returns whether Inst is compressed.
  bool EmitToStreamer(MCStreamer &S, const MCInst &Inst,
                      const MCSubtargetInfo &SubtargetInfo);
  bool EmitToStreamer(MCStreamer &S, const MCInst &Inst) {
    return EmitToStreamer(S, Inst, *STI);
  }

  bool lowerPseudoInstExpansion(const MachineInstr *MI, MCInst &Inst);

  typedef std::tuple<unsigned, uint32_t> HwasanMemaccessTuple;
  std::map<HwasanMemaccessTuple, MCSymbol *> HwasanMemaccessSymbols;
  void LowerHWASAN_CHECK_MEMACCESS(const MachineInstr &MI);
  void LowerKCFI_CHECK(const MachineInstr &MI);
  void EmitHwasanMemaccessSymbols(Module &M);

  // Wrapper needed for tblgenned pseudo lowering.
  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp) const;

  void emitStartOfAsmFile(Module &M) override;
  void emitEndOfAsmFile(Module &M) override;

  void emitFunctionEntryLabel() override;
  bool emitDirectiveOptionArch();

  void emitNoteGnuProperty(const Module &M);

private:
  void emitAttributes(const MCSubtargetInfo &SubtargetInfo);

  void emitNTLHint(const MachineInstr *MI);

  // XRay Support
  void LowerPATCHABLE_FUNCTION_ENTER(const MachineInstr *MI);
  void LowerPATCHABLE_FUNCTION_EXIT(const MachineInstr *MI);
  void LowerPATCHABLE_TAIL_CALL(const MachineInstr *MI);
  void emitSled(const MachineInstr *MI, SledKind Kind);

  bool lowerToMCInst(const MachineInstr *MI, MCInst &OutMI);
};
}

// A COGBT guest-memory record is paired with the access it describes, so the
// access must be a single scalar load or store with one memory operand.  The
// AOT runtime never decodes the faulting opcode, and a floating-point access
// names an FP register, which a record built from i64 operands never denotes,
// so scalar FP accesses are exactly as recoverable as the integer ones.
static bool isCOGBTRecordableGuestAccess(const MachineInstr &MI) {
  // One instruction may carry more than one volatile memory operand, e.g. a
  // single store whose source pair names two guest locations.  The record is
  // still recoverable: the runtime matches it by exact offset, restores the
  // recorded guest state and returns to the interpreter to re-execute the
  // instruction, so every operand of that instruction is handled.  Accept such
  // an access only when it has at least one memory operand and all of them are
  // volatile and non-atomic; the opcode fixed by the switch below gives the
  // access width.
  if (MI.memoperands_empty())
    return false;
  for (const MachineMemOperand *MMO : MI.memoperands())
    if (!MMO->isVolatile() || MMO->isAtomic())
      return false;
  switch (MI.getOpcode()) {
  case RISCV::LB:
  case RISCV::LBU:
  case RISCV::LH:
  case RISCV::LHU:
  case RISCV::LW:
  case RISCV::LWU:
  case RISCV::LD:
  case RISCV::SB:
  case RISCV::SH:
  case RISCV::SW:
  case RISCV::SD:
  case RISCV::FLH:
  case RISCV::FLW:
  case RISCV::FLD:
  case RISCV::FSH:
  case RISCV::FSW:
  case RISCV::FSD:
    return true;
  default:
    return false;
  }
}

static bool isCOGBTGuestLoad(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case RISCV::LB:
  case RISCV::LBU:
  case RISCV::LH:
  case RISCV::LHU:
  case RISCV::LW:
  case RISCV::LWU:
  case RISCV::LD:
  case RISCV::FLH:
  case RISCV::FLW:
  case RISCV::FLD:
    return true;
  default:
    return false;
  }
}

static bool isCOGBTLoadDestination(const MachineInstr &Access, Register Reg,
                                   const TargetRegisterInfo &TRI) {
  if (!isCOGBTGuestLoad(Access) || Access.getNumOperands() == 0 ||
      !Access.getOperand(0).isReg())
    return false;
  Register Destination = Access.getOperand(0).getReg();
  return Destination.isPhysical() && TRI.regsOverlap(Destination, Reg);
}

static bool isCOGBTRecordedRegisterOverwritten(const MachineInstr &Access,
                                               Register Reg,
                                               const TargetRegisterInfo &TRI) {
  return Access.modifiesRegister(Reg, &TRI) &&
         !TRI.isConstantPhysReg(Reg) &&
         !isCOGBTLoadDestination(Access, Reg, TRI);
}

// A scalar spill into this function's own frame is invisible to the guest: it
// writes host stack memory rather than guest memory, and it defines no
// register, so it cannot invalidate a recorded location.  A recorded value's
// own spill slot cannot be reused for something else before the access either,
// because the translator keeps every recorded value live past the access.
// Name the translation unit, the record and the offending instruction so a
// rejected build points straight at the machine code that has to change.
[[noreturn]] static void reportCOGBTStackmapError(const char *Reason,
                                                  const MachineInstr &Stackmap,
                                                  const MachineInstr *Offender) {
  std::string Message;
  raw_string_ostream OS(Message);
  OS << Reason << " in " << Stackmap.getMF()->getName() << "\n  record: ";
  Stackmap.print(OS, /*IsStandalone=*/false, /*SkipOpers=*/false,
                 /*SkipDebugLoc=*/true, /*AddNewLine=*/false);
  if (Offender) {
    OS << "\n  offending instruction: ";
    Offender->print(OS, /*IsStandalone=*/false, /*SkipOpers=*/false,
                    /*SkipDebugLoc=*/true, /*AddNewLine=*/false);
  }
  if (StringRef(Reason) == "COGBT stackmap is missing access-point fake uses" ||
      StringRef(Reason) ==
          "COGBT stackmap has no adjacent guest memory instruction" ||
      StringRef(Reason) ==
          "COGBT stackmap location is defined before its guest access" ||
      StringRef(Reason) ==
          "COGBT guest access overwrites a recorded register") {
    OS << "\n  machine function:\n";
    Stackmap.getMF()->print(OS);
  }
  report_fatal_error(Twine(Message));
}

// Walk forward from the start of \p B looking for the guest access a record
// describes, following side-effect-free unconditional branches the same way
// the adjacency scan does.  Returns null when no access is reachable within
// the budget; the caller decides what that means.
static const MachineInstr *findCOGBTGuestAccessAhead(const MachineBasicBlock *B,
                                                     unsigned Budget) {
  for (; B && Budget > 0; --Budget) {
    bool Hopped = false;
    for (auto It = B->begin(); It != B->end(); ++It) {
      if (It->isMetaInstruction())
        continue;
      bool VolatileAccess = false;
      for (const MachineMemOperand *MMO : It->memoperands())
        VolatileAccess |= MMO->isVolatile();
      if (VolatileAccess)
        return &*It;
      if (It->isTerminator()) {
        B = (It->isUnconditionalBranch() && !It->hasUnmodeledSideEffects())
                ? B->getSingleSuccessor()
                : nullptr;
        Hopped = true;
        break;
      }
    }
    if (!Hopped)
      return nullptr;
  }
  return nullptr;
}

static bool isCOGBTHostFrameSpill(const MachineInstr &MI) {
  if (!MI.hasOneMemOperand())
    return false;
  switch (MI.getOpcode()) {
  case RISCV::SB:
  case RISCV::SH:
  case RISCV::SW:
  case RISCV::SD:
  case RISCV::FSH:
  case RISCV::FSW:
  case RISCV::FSD:
    break;
  default:
    return false;
  }
  const MachineMemOperand *MMO = *MI.memoperands_begin();
  if (!MMO->isStore() || MMO->isVolatile() || MMO->isAtomic())
    return false;
  const PseudoSourceValue *PSV = MMO->getPseudoValue();
  return PSV && (PSV->kind() == PseudoSourceValue::FixedStack ||
                 PSV->kind() == PseudoSourceValue::Stack);
}

struct COGBTFakeUsePath {
  SmallVector<Register, 8> Registers;
  SmallVector<unsigned, 8> InterveningCounts;
  SmallVector<const MachineInstr *, 8> InterveningInstructions;
};

static bool collectCOGBTFakeUses(
    const MachineBasicBlock &MBB, MachineBasicBlock::const_iterator Next,
    unsigned Required, const TargetRegisterInfo &TRI,
    SmallPtrSetImpl<const MachineBasicBlock *> &Visited,
    COGBTFakeUsePath &Path) {
  for (; Next != MBB.end() && Path.Registers.size() < Required; ++Next) {
    if (Next->isFakeUse()) {
      if (Next->getNumOperands() != 1 || !Next->getOperand(0).isReg() ||
          !Next->getOperand(0).getReg().isPhysical() ||
          Next->getOperand(0).getSubReg())
        return false;
      Register Reg = Next->getOperand(0).getReg();
      if (TRI.isConstantPhysReg(Reg))
        continue;
      Path.Registers.push_back(Reg);
      Path.InterveningCounts.push_back(Path.InterveningInstructions.size());
      continue;
    }
    if (Next->isMetaInstruction())
      continue;
    if (Next->getOpcode() == TargetOpcode::STACKMAP)
      return false;
    Path.InterveningInstructions.push_back(&*Next);
  }
  if (Path.Registers.size() == Required)
    return true;

  for (const MachineBasicBlock *Successor : MBB.successors()) {
    if (!Visited.insert(Successor).second)
      continue;
    COGBTFakeUsePath SuccessorPath = Path;
    if (collectCOGBTFakeUses(*Successor, Successor->begin(), Required, TRI,
                             Visited, SuccessorPath)) {
      Path = std::move(SuccessorPath);
      return true;
    }
    Visited.erase(Successor);
  }
  return false;
}

// The translator emits one llvm.fake.use immediately after the guest access
// for every non-constant stackmap operand.  Register allocation can split a
// value's live range between the stackmap and that access, so the original
// STACKMAP operand may name a register that has already been reused.  The
// FAKE_USE operand names the same value after the access.  Retarget register
// locations to those physical registers, while proving that no instruction
// between the access and each fake use defines that register.  Unrelated
// instructions may be scheduled in between; a reload or clobber of the chosen
// register would mean it was not yet valid when the access faulted.
static void retargetCOGBTStackmapRegisters(MachineInstr &Stackmap,
                                           const MachineInstr &Access) {
  const TargetRegisterInfo *TRI =
      Stackmap.getMF()->getSubtarget().getRegisterInfo();
  SmallVector<Register, 8> RecordedRegisters;
  unsigned NonConstantOperands = 0;
  for (unsigned Index = StackMapOpers(&Stackmap).getVarIdx();
       Index < Stackmap.getNumOperands();) {
    const MachineOperand &Operand = Stackmap.getOperand(Index);
    if (Operand.isImplicit()) {
      ++Index;
      continue;
    }
    if (Operand.isReg()) {
      if (!Operand.getReg().isPhysical() || Operand.getSubReg())
        reportCOGBTStackmapError(
            "COGBT stackmap has an unsupported register location", Stackmap,
            nullptr);
      RecordedRegisters.push_back(Operand.getReg());
      ++NonConstantOperands;
      ++Index;
      continue;
    }
    if (!Operand.isImm())
      reportCOGBTStackmapError("COGBT stackmap has an unsupported operand",
                               Stackmap, nullptr);
    switch (Operand.getImm()) {
    case StackMaps::DirectMemRefOp:
      if (Index + 2 >= Stackmap.getNumOperands())
        reportCOGBTStackmapError("COGBT stackmap has a truncated direct location",
                                 Stackmap, nullptr);
      if (!Stackmap.getOperand(Index + 1).isReg() ||
          !Stackmap.getOperand(Index + 1).getReg().isPhysical() ||
          Stackmap.getOperand(Index + 1).getSubReg())
        reportCOGBTStackmapError(
            "COGBT stackmap has an unsupported direct base register", Stackmap,
            nullptr);
      RecordedRegisters.push_back(Stackmap.getOperand(Index + 1).getReg());
      ++NonConstantOperands;
      Index += 3;
      break;
    case StackMaps::IndirectMemRefOp:
      if (Index + 3 >= Stackmap.getNumOperands())
        reportCOGBTStackmapError(
            "COGBT stackmap has a truncated indirect location", Stackmap,
            nullptr);
      if (!Stackmap.getOperand(Index + 2).isReg() ||
          !Stackmap.getOperand(Index + 2).getReg().isPhysical() ||
          Stackmap.getOperand(Index + 2).getSubReg())
        reportCOGBTStackmapError(
            "COGBT stackmap has an unsupported indirect base register",
            Stackmap, nullptr);
      RecordedRegisters.push_back(Stackmap.getOperand(Index + 2).getReg());
      ++NonConstantOperands;
      Index += 4;
      break;
    case StackMaps::ConstantOp:
      if (Index + 1 >= Stackmap.getNumOperands())
        reportCOGBTStackmapError("COGBT stackmap has a truncated constant",
                                 Stackmap, nullptr);
      Index += 2;
      break;
    default:
      reportCOGBTStackmapError("COGBT stackmap has an unknown operand tag",
                               Stackmap, nullptr);
    }
  }

  SmallVector<Register, 8> AccessRegisters;
  COGBTFakeUsePath FakeUsePath;
  SmallPtrSet<const MachineBasicBlock *, 8> Visited;
  Visited.insert(Access.getParent());
  bool HasFakeUsePath =
      collectCOGBTFakeUses(*Access.getParent(),
                           std::next(Access.getIterator()),
                           NonConstantOperands, *TRI, Visited, FakeUsePath);
  if (!HasFakeUsePath) {
    // Some legal vector-lowered values have no machine FAKE_USE after the
    // access: their scalar chunks are folded into constants or the target
    // DAG cannot materialize the original extract value.  In that case the
    // original stackmap locations remain exact if no instruction can define
    // a recorded register between the record and its adjacent access.
    if (Stackmap.getParent() != Access.getParent())
      reportCOGBTStackmapError(
          "COGBT stackmap is missing access-point fake uses", Stackmap,
          &Access);
    for (auto It = std::next(Stackmap.getIterator()); It != Access.getIterator();
         ++It) {
      if (It->isMetaInstruction())
        continue;
      for (Register Reg : RecordedRegisters)
        if (It->modifiesRegister(Reg, TRI))
          reportCOGBTStackmapError(
              "COGBT stackmap location is defined before its guest access",
              Stackmap, &*It);
    }
    for (Register Reg : RecordedRegisters)
      if (isCOGBTRecordedRegisterOverwritten(Access, Reg, *TRI))
        reportCOGBTStackmapError(
            "COGBT guest access overwrites a recorded register", Stackmap,
            &Access);
    return;
  }
  AccessRegisters = FakeUsePath.Registers;

  for (unsigned I = 0; I < AccessRegisters.size(); ++I)
    for (unsigned J = 0; J < FakeUsePath.InterveningCounts[I]; ++J)
      if (FakeUsePath.InterveningInstructions[J]->modifiesRegister(
              AccessRegisters[I], TRI))
        reportCOGBTStackmapError(
            "COGBT access-point register is defined after the guest access",
            Stackmap, FakeUsePath.InterveningInstructions[J]);

  unsigned AccessRegisterIndex = 0;
  for (unsigned Index = StackMapOpers(&Stackmap).getVarIdx();
       Index < Stackmap.getNumOperands();) {
    MachineOperand &Operand = Stackmap.getOperand(Index);
    if (Operand.isImplicit()) {
      ++Index;
      continue;
    }
    if (Operand.isReg()) {
      Register AccessReg = AccessRegisters[AccessRegisterIndex++];
      if (isCOGBTRecordedRegisterOverwritten(Access, AccessReg, *TRI))
        reportCOGBTStackmapError(
            "COGBT guest access overwrites a recorded register", Stackmap,
            &Access);
      Operand.setReg(AccessReg);
      ++Index;
      continue;
    }
    switch (Operand.getImm()) {
    case StackMaps::DirectMemRefOp:
      ++AccessRegisterIndex;
      Index += 3;
      break;
    case StackMaps::IndirectMemRefOp:
      ++AccessRegisterIndex;
      Index += 4;
      break;
    case StackMaps::ConstantOp:
      Index += 2;
      break;
    default:
      llvm_unreachable("validated COGBT stackmap operand changed");
    }
  }
  assert(AccessRegisterIndex == AccessRegisters.size() &&
         "failed to consume COGBT access-point registers");
}

void RISCVAsmPrinter::LowerSTACKMAP(MCStreamer &OutStreamer, StackMaps &SM,
                                    const MachineInstr &MI) {
  unsigned NOPBytes = STI->hasStdExtZca() ? 2 : 4;
  unsigned NumNOPBytes = StackMapOpers(&MI).getNumPatchBytes();

  if (StackMapOpers(&MI).getID() == 99999 && NumNOPBytes == 0 &&
      MI.getMF()->getFunction().hasFnAttribute("cogbt-guest-memory-stackmap")) {
    const TargetRegisterInfo *TRI = STI->getRegisterInfo();
    // The record and the access it describes are emitted from the same
    // machine function, but a side-effect-free unconditional branch may sit
    // between them: the access then lives in the successor block.  Walk such
    // branches (bounded) instead of rejecting the whole build, and keep
    // rejecting every other intervening instruction that could change
    // guest-visible state.
    const MachineBasicBlock *ScanBlock = MI.getParent();
    MachineBasicBlock::const_iterator Next = std::next(MI.getIterator());
    unsigned BlockBudget = 64;
    for (;;) {
      bool Hopped = false;
      for (; Next != ScanBlock->end(); ++Next) {
        if (Next->isMetaInstruction())
          continue;
        bool VolatileAccess = false;
        for (const MachineMemOperand *MMO : Next->memoperands())
          VolatileAccess |= MMO->isVolatile();
        if (VolatileAccess) {
          if (!isCOGBTRecordableGuestAccess(*Next))
            reportCOGBTStackmapError(
                "COGBT stackmap target is not a scalar memory instruction", MI,
                &*Next);
          retargetCOGBTStackmapRegisters(const_cast<MachineInstr &>(MI), *Next);
          for (const auto &Pending : PendingGuestStackMaps) {
            if (Pending.second != &*Next)
              continue;
            // Two records can pair with the same access when several paths
            // reach one instruction (or when the access instruction carries
            // several guest locations).  The access describes one guest
            // access point, so keep the first record and drop the duplicate
            // instead of failing the whole build.
            return;
          }
          PendingGuestStackMaps.push_back(std::make_pair(&MI, &*Next));
          return;
        }
        // Any terminator ends the basic block: PseudoBR, for instance, is a
        // terminator without the branch flag, and treating it as an ordinary
        // instruction let the scan run off the end of the block.
        if (Next->isTerminator()) {
          const MachineBasicBlock *Succ = nullptr;
          const bool Hoppable = !Next->hasUnmodeledSideEffects() &&
                                !Next->modifiesRegister(RISCV::X2, TRI) &&
                                BlockBudget > 0;
          if (Hoppable && Next->isUnconditionalBranch()) {
            Succ = ScanBlock->getSingleSuccessor();
          } else if (Hoppable) {
            // A conditional (or otherwise multi-way) terminator may still sit
            // between the record and its access: follow it only when exactly
            // one successor actually leads to a guest access.
            for (const MachineBasicBlock *Candidate : ScanBlock->successors()) {
              if (!findCOGBTGuestAccessAhead(Candidate, 64))
                continue;
              if (Succ) {
                Succ = nullptr;
                break;
              }
              Succ = Candidate;
            }
          }
          if (!Succ)
            reportCOGBTStackmapError(
                "COGBT stackmap crosses a guest-state side effect", MI, &*Next);
          --BlockBudget;
          ScanBlock = Succ;
          Next = Succ->begin();
          Hopped = true;
          break;
        }
        if ((Next->mayStore() && !isCOGBTHostFrameSpill(*Next)) ||
            Next->isCall() || Next->hasUnmodeledSideEffects() ||
            Next->modifiesRegister(RISCV::X2, TRI))
          reportCOGBTStackmapError(
              "COGBT stackmap crosses a guest-state side effect", MI, &*Next);
      }
      if (!Hopped) {
        // The block can also end without a terminator we recognise (an
        // implicit fallthrough into the layout successor); follow it the same
        // way before giving up.
        const MachineBasicBlock *FallThrough =
            const_cast<MachineBasicBlock *>(ScanBlock)->getFallThrough();
        if (FallThrough && BlockBudget > 0) {
          --BlockBudget;
          ScanBlock = FallThrough;
          Next = FallThrough->begin();
          continue;
        }
        reportCOGBTStackmapError(
            "COGBT stackmap has no adjacent guest memory instruction", MI,
            nullptr);
      }
    }
  }

  auto &Ctx = OutStreamer.getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer.emitLabel(MILabel);

  SM.recordStackMap(*MILabel, MI);
  assert(NumNOPBytes % NOPBytes == 0 &&
         "Invalid number of NOP bytes requested!");

  // Scan ahead to trim the shadow.
  const MachineBasicBlock &MBB = *MI.getParent();
  MachineBasicBlock::const_iterator MII(MI);
  ++MII;
  while (NumNOPBytes > 0) {
    if (MII == MBB.end() || MII->isCall() ||
        MII->getOpcode() == RISCV::DBG_VALUE ||
        MII->getOpcode() == TargetOpcode::PATCHPOINT ||
        MII->getOpcode() == TargetOpcode::STACKMAP)
      break;
    ++MII;
    NumNOPBytes -= 4;
  }

  // Emit nops.
  emitNops(NumNOPBytes / NOPBytes);
}

// Lower a patchpoint of the form:
// [<def>], <id>, <numBytes>, <target>, <numArgs>
void RISCVAsmPrinter::LowerPATCHPOINT(MCStreamer &OutStreamer, StackMaps &SM,
                                      const MachineInstr &MI) {
  unsigned NOPBytes = STI->hasStdExtZca() ? 2 : 4;

  auto &Ctx = OutStreamer.getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer.emitLabel(MILabel);
  SM.recordPatchPoint(*MILabel, MI);

  PatchPointOpers Opers(&MI);

  const MachineOperand &CalleeMO = Opers.getCallTarget();
  unsigned EncodedBytes = 0;

  if (CalleeMO.isImm()) {
    uint64_t CallTarget = CalleeMO.getImm();
    if (CallTarget) {
      assert((CallTarget & 0xFFFF'FFFF'FFFF) == CallTarget &&
             "High 16 bits of call target should be zero.");
      // Materialize the jump address:
      SmallVector<MCInst, 8> Seq;
      RISCVMatInt::generateMCInstSeq(CallTarget, *STI, RISCV::X1, Seq);
      for (MCInst &Inst : Seq) {
        bool Compressed = EmitToStreamer(OutStreamer, Inst);
        EncodedBytes += Compressed ? 2 : 4;
      }
      bool Compressed = EmitToStreamer(OutStreamer, MCInstBuilder(RISCV::JALR)
                                                        .addReg(RISCV::X1)
                                                        .addReg(RISCV::X1)
                                                        .addImm(0));
      EncodedBytes += Compressed ? 2 : 4;
    }
  } else if (CalleeMO.isGlobal()) {
    MCOperand CallTargetMCOp;
    lowerOperand(CalleeMO, CallTargetMCOp);
    EmitToStreamer(OutStreamer,
                   MCInstBuilder(RISCV::PseudoCALL).addOperand(CallTargetMCOp));
    EncodedBytes += 8;
  }

  // Emit padding.
  unsigned NumBytes = Opers.getNumPatchBytes();
  assert(NumBytes >= EncodedBytes &&
         "Patchpoint can't request size less than the length of a call.");
  assert((NumBytes - EncodedBytes) % NOPBytes == 0 &&
         "Invalid number of NOP bytes requested!");
  emitNops((NumBytes - EncodedBytes) / NOPBytes);
}

void RISCVAsmPrinter::LowerSTATEPOINT(MCStreamer &OutStreamer, StackMaps &SM,
                                      const MachineInstr &MI) {
  unsigned NOPBytes = STI->hasStdExtZca() ? 2 : 4;

  StatepointOpers SOpers(&MI);
  if (unsigned PatchBytes = SOpers.getNumPatchBytes()) {
    assert(PatchBytes % NOPBytes == 0 &&
           "Invalid number of NOP bytes requested!");
    emitNops(PatchBytes / NOPBytes);
  } else {
    // Lower call target and choose correct opcode
    const MachineOperand &CallTarget = SOpers.getCallTarget();
    MCOperand CallTargetMCOp;
    switch (CallTarget.getType()) {
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_ExternalSymbol:
      lowerOperand(CallTarget, CallTargetMCOp);
      EmitToStreamer(
          OutStreamer,
          MCInstBuilder(RISCV::PseudoCALL).addOperand(CallTargetMCOp));
      break;
    case MachineOperand::MO_Immediate:
      CallTargetMCOp = MCOperand::createImm(CallTarget.getImm());
      EmitToStreamer(OutStreamer, MCInstBuilder(RISCV::JAL)
                                      .addReg(RISCV::X1)
                                      .addOperand(CallTargetMCOp));
      break;
    case MachineOperand::MO_Register:
      CallTargetMCOp = MCOperand::createReg(CallTarget.getReg());
      EmitToStreamer(OutStreamer, MCInstBuilder(RISCV::JALR)
                                      .addReg(RISCV::X1)
                                      .addOperand(CallTargetMCOp)
                                      .addImm(0));
      break;
    default:
      llvm_unreachable("Unsupported operand type in statepoint call target");
      break;
    }
  }

  auto &Ctx = OutStreamer.getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer.emitLabel(MILabel);
  SM.recordStatepoint(*MILabel, MI);
}

bool RISCVAsmPrinter::EmitToStreamer(MCStreamer &S, const MCInst &Inst,
                                     const MCSubtargetInfo &SubtargetInfo) {
  MCInst CInst;
  bool Res = RISCVRVC::compress(CInst, Inst, SubtargetInfo);
  if (Res)
    ++RISCVNumInstrsCompressed;
  S.emitInstruction(Res ? CInst : Inst, SubtargetInfo);
  return Res;
}

// Simple pseudo-instructions have their lowering (with expansion to real
// instructions) auto-generated.
#include "RISCVGenMCPseudoLowering.inc"

// If the target supports Zihintntl and the instruction has a nontemporal
// MachineMemOperand, emit an NTLH hint instruction before it.
void RISCVAsmPrinter::emitNTLHint(const MachineInstr *MI) {
  if (!STI->hasStdExtZihintntl())
    return;

  if (MI->memoperands_empty())
    return;

  MachineMemOperand *MMO = *(MI->memoperands_begin());
  if (!MMO->isNonTemporal())
    return;

  unsigned NontemporalMode = 0;
  if (MMO->getFlags() & MONontemporalBit0)
    NontemporalMode += 0b1;
  if (MMO->getFlags() & MONontemporalBit1)
    NontemporalMode += 0b10;

  MCInst Hint;
  if (STI->hasStdExtZca())
    Hint.setOpcode(RISCV::C_ADD_HINT);
  else
    Hint.setOpcode(RISCV::ADD);

  Hint.addOperand(MCOperand::createReg(RISCV::X0));
  Hint.addOperand(MCOperand::createReg(RISCV::X0));
  Hint.addOperand(MCOperand::createReg(RISCV::X2 + NontemporalMode));

  EmitToStreamer(*OutStreamer, Hint);
}

void RISCVAsmPrinter::emitInstruction(const MachineInstr *MI) {
  RISCV_MC::verifyInstructionPredicates(MI->getOpcode(), STI->getFeatureBits());

  emitNTLHint(MI);

  for (size_t I = 0; I < PendingGuestStackMaps.size();) {
    if (PendingGuestStackMaps[I].second != MI) {
      ++I;
      continue;
    }
    MCSymbol *Label = OutStreamer->getContext().createTempSymbol();
    OutStreamer->emitLabel(Label);
    SM.recordStackMap(*Label, *PendingGuestStackMaps[I].first);
    PendingGuestStackMaps.erase(PendingGuestStackMaps.begin() + I);
  }

  // Do any auto-generated pseudo lowerings.
  if (MCInst OutInst; lowerPseudoInstExpansion(MI, OutInst)) {
    EmitToStreamer(*OutStreamer, OutInst);
    return;
  }

  switch (MI->getOpcode()) {
  case RISCV::HWASAN_CHECK_MEMACCESS_SHORTGRANULES:
    LowerHWASAN_CHECK_MEMACCESS(*MI);
    return;
  case RISCV::KCFI_CHECK:
    LowerKCFI_CHECK(*MI);
    return;
  case TargetOpcode::STACKMAP:
    return LowerSTACKMAP(*OutStreamer, SM, *MI);
  case TargetOpcode::PATCHPOINT:
    return LowerPATCHPOINT(*OutStreamer, SM, *MI);
  case TargetOpcode::STATEPOINT:
    return LowerSTATEPOINT(*OutStreamer, SM, *MI);
  case TargetOpcode::PATCHABLE_FUNCTION_ENTER: {
    // patchable-function-entry is handled in lowerToMCInst
    // Therefore, we break out of the switch statement if we encounter it here.
    const Function &F = MI->getParent()->getParent()->getFunction();
    if (F.hasFnAttribute("patchable-function-entry"))
      break;

    LowerPATCHABLE_FUNCTION_ENTER(MI);
    return;
  }
  case TargetOpcode::PATCHABLE_FUNCTION_EXIT:
    LowerPATCHABLE_FUNCTION_EXIT(MI);
    return;
  case TargetOpcode::PATCHABLE_TAIL_CALL:
    LowerPATCHABLE_TAIL_CALL(MI);
    return;
  }

  MCInst OutInst;
  if (!lowerToMCInst(MI, OutInst))
    EmitToStreamer(*OutStreamer, OutInst);
}

bool RISCVAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                      const char *ExtraCode, raw_ostream &OS) {
  // First try the generic code, which knows about modifiers like 'c' and 'n'.
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS))
    return false;

  const MachineOperand &MO = MI->getOperand(OpNo);
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0)
      return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      return true; // Unknown modifier.
    case 'z':      // Print zero register if zero, regular printing otherwise.
      if (MO.isImm() && MO.getImm() == 0) {
        OS << RISCVInstPrinter::getRegisterName(RISCV::X0);
        return false;
      }
      break;
    case 'i': // Literal 'i' if operand is not a register.
      if (!MO.isReg())
        OS << 'i';
      return false;
    case 'N': // Print the register encoding as an integer (0-31)
      if (!MO.isReg())
        return true;

      const RISCVRegisterInfo *TRI = STI->getRegisterInfo();
      OS << TRI->getEncodingValue(MO.getReg());
      return false;
    }
  }

  switch (MO.getType()) {
  case MachineOperand::MO_Immediate:
    OS << MO.getImm();
    return false;
  case MachineOperand::MO_Register:
    OS << RISCVInstPrinter::getRegisterName(MO.getReg());
    return false;
  case MachineOperand::MO_GlobalAddress:
    PrintSymbolOperand(MO, OS);
    return false;
  case MachineOperand::MO_BlockAddress: {
    MCSymbol *Sym = GetBlockAddressSymbol(MO.getBlockAddress());
    Sym->print(OS, MAI);
    return false;
  }
  default:
    break;
  }

  return true;
}

bool RISCVAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                            unsigned OpNo,
                                            const char *ExtraCode,
                                            raw_ostream &OS) {
  if (ExtraCode)
    return AsmPrinter::PrintAsmMemoryOperand(MI, OpNo, ExtraCode, OS);

  const MachineOperand &AddrReg = MI->getOperand(OpNo);
  assert(MI->getNumOperands() > OpNo + 1 && "Expected additional operand");
  const MachineOperand &Offset = MI->getOperand(OpNo + 1);
  // All memory operands should have a register and an immediate operand (see
  // RISCVDAGToDAGISel::SelectInlineAsmMemoryOperand).
  if (!AddrReg.isReg())
    return true;
  if (!Offset.isImm() && !Offset.isGlobal() && !Offset.isBlockAddress() &&
      !Offset.isMCSymbol())
    return true;

  MCOperand MCO;
  if (!lowerOperand(Offset, MCO))
    return true;

  if (Offset.isImm())
    OS << MCO.getImm();
  else if (Offset.isGlobal() || Offset.isBlockAddress() || Offset.isMCSymbol())
    MAI->printExpr(OS, *MCO.getExpr());

  if (Offset.isMCSymbol())
    MMI->getContext().registerInlineAsmLabel(Offset.getMCSymbol());
  if (Offset.isBlockAddress()) {
    const BlockAddress *BA = Offset.getBlockAddress();
    MCSymbol *Sym = GetBlockAddressSymbol(BA);
    MMI->getContext().registerInlineAsmLabel(Sym);
  }

  OS << "(" << RISCVInstPrinter::getRegisterName(AddrReg.getReg()) << ")";
  return false;
}

bool RISCVAsmPrinter::emitDirectiveOptionArch() {
  RISCVTargetStreamer &RTS =
      static_cast<RISCVTargetStreamer &>(*OutStreamer->getTargetStreamer());
  SmallVector<RISCVOptionArchArg> NeedEmitStdOptionArgs;
  const MCSubtargetInfo &MCSTI = *TM.getMCSubtargetInfo();
  for (const auto &Feature : RISCVFeatureKV) {
    if (STI->hasFeature(Feature.Value) == MCSTI.hasFeature(Feature.Value))
      continue;

    if (!llvm::RISCVISAInfo::isSupportedExtensionFeature(Feature.Key))
      continue;

    auto Delta = STI->hasFeature(Feature.Value) ? RISCVOptionArchArgType::Plus
                                                : RISCVOptionArchArgType::Minus;
    NeedEmitStdOptionArgs.emplace_back(Delta, Feature.Key);
  }
  if (!NeedEmitStdOptionArgs.empty()) {
    RTS.emitDirectiveOptionPush();
    RTS.emitDirectiveOptionArch(NeedEmitStdOptionArgs);
    return true;
  }

  return false;
}

bool RISCVAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  if (!PendingGuestStackMaps.empty())
    report_fatal_error("COGBT stackmap target was not emitted");
  STI = &MF.getSubtarget<RISCVSubtarget>();
  RISCVTargetStreamer &RTS =
      static_cast<RISCVTargetStreamer &>(*OutStreamer->getTargetStreamer());

  bool EmittedOptionArch = emitDirectiveOptionArch();

  SetupMachineFunction(MF);
  emitFunctionBody();
  if (!PendingGuestStackMaps.empty())
    report_fatal_error("COGBT stackmap target was not emitted");

  // Emit the XRay table
  emitXRayTable();

  if (EmittedOptionArch)
    RTS.emitDirectiveOptionPop();
  return false;
}

void RISCVAsmPrinter::LowerPATCHABLE_FUNCTION_ENTER(const MachineInstr *MI) {
  emitSled(MI, SledKind::FUNCTION_ENTER);
}

void RISCVAsmPrinter::LowerPATCHABLE_FUNCTION_EXIT(const MachineInstr *MI) {
  emitSled(MI, SledKind::FUNCTION_EXIT);
}

void RISCVAsmPrinter::LowerPATCHABLE_TAIL_CALL(const MachineInstr *MI) {
  emitSled(MI, SledKind::TAIL_CALL);
}

void RISCVAsmPrinter::emitSled(const MachineInstr *MI, SledKind Kind) {
  // We want to emit the jump instruction and the nops constituting the sled.
  // The format is as follows:
  // .Lxray_sled_N
  //   ALIGN
  //   J .tmpN
  //   21 or 33 C.NOP instructions
  // .tmpN

  // The following variable holds the count of the number of NOPs to be patched
  // in for XRay instrumentation during compilation.
  // Note that RV64 and RV32 each has a sled of 68 and 44 bytes, respectively.
  // Assuming we're using JAL to jump to .tmpN, then we only need
  // (68 - 4)/2 = 32 NOPs for RV64 and (44 - 4)/2 = 20 for RV32. However, there
  // is a chance that we'll use C.JAL instead, so an additional NOP is needed.
  const uint8_t NoopsInSledCount = STI->is64Bit() ? 33 : 21;

  OutStreamer->emitCodeAlignment(Align(4), STI);
  auto CurSled = OutContext.createTempSymbol("xray_sled_", true);
  OutStreamer->emitLabel(CurSled);
  auto Target = OutContext.createTempSymbol();

  const MCExpr *TargetExpr = MCSymbolRefExpr::create(Target, OutContext);

  // Emit "J bytes" instruction, which jumps over the nop sled to the actual
  // start of function.
  EmitToStreamer(
      *OutStreamer,
      MCInstBuilder(RISCV::JAL).addReg(RISCV::X0).addExpr(TargetExpr));

  // Emit NOP instructions
  for (int8_t I = 0; I < NoopsInSledCount; ++I)
    EmitToStreamer(*OutStreamer, MCInstBuilder(RISCV::ADDI)
                                     .addReg(RISCV::X0)
                                     .addReg(RISCV::X0)
                                     .addImm(0));

  OutStreamer->emitLabel(Target);
  recordSled(CurSled, *MI, Kind, 2);
}

void RISCVAsmPrinter::emitStartOfAsmFile(Module &M) {
  RISCVTargetStreamer &RTS =
      static_cast<RISCVTargetStreamer &>(*OutStreamer->getTargetStreamer());
  if (const MDString *ModuleTargetABI =
          dyn_cast_or_null<MDString>(M.getModuleFlag("target-abi")))
    RTS.setTargetABI(RISCVABI::getTargetABI(ModuleTargetABI->getString()));

  MCSubtargetInfo SubtargetInfo = *TM.getMCSubtargetInfo();

  // Use module flag to update feature bits.
  if (auto *MD = dyn_cast_or_null<MDNode>(M.getModuleFlag("riscv-isa"))) {
    for (auto &ISA : MD->operands()) {
      if (auto *ISAString = dyn_cast_or_null<MDString>(ISA)) {
        auto ParseResult = llvm::RISCVISAInfo::parseArchString(
            ISAString->getString(), /*EnableExperimentalExtension=*/true,
            /*ExperimentalExtensionVersionCheck=*/true);
        if (!errorToBool(ParseResult.takeError())) {
          auto &ISAInfo = *ParseResult;
          for (const auto &Feature : RISCVFeatureKV) {
            if (ISAInfo->hasExtension(Feature.Key) &&
                !SubtargetInfo.hasFeature(Feature.Value))
              SubtargetInfo.ToggleFeature(Feature.Key);
          }
        }
      }
    }

    RTS.setFlagsFromFeatures(SubtargetInfo);
  }

  if (TM.getTargetTriple().isOSBinFormatELF())
    emitAttributes(SubtargetInfo);
}

void RISCVAsmPrinter::emitEndOfAsmFile(Module &M) {
  RISCVTargetStreamer &RTS =
      static_cast<RISCVTargetStreamer &>(*OutStreamer->getTargetStreamer());

  if (TM.getTargetTriple().isOSBinFormatELF()) {
    RTS.finishAttributeSection();
    emitNoteGnuProperty(M);
  }
  EmitHwasanMemaccessSymbols(M);
}

void RISCVAsmPrinter::emitAttributes(const MCSubtargetInfo &SubtargetInfo) {
  RISCVTargetStreamer &RTS =
      static_cast<RISCVTargetStreamer &>(*OutStreamer->getTargetStreamer());
  // Use MCSubtargetInfo from TargetMachine. Individual functions may have
  // attributes that differ from other functions in the module and we have no
  // way to know which function is correct.
  RTS.emitTargetAttributes(SubtargetInfo, /*EmitStackAlign*/ true);
}

void RISCVAsmPrinter::emitFunctionEntryLabel() {
  const auto *RMFI = MF->getInfo<RISCVMachineFunctionInfo>();
  if (RMFI->isVectorCall()) {
    auto &RTS =
        static_cast<RISCVTargetStreamer &>(*OutStreamer->getTargetStreamer());
    RTS.emitDirectiveVariantCC(*CurrentFnSym);
  }
  return AsmPrinter::emitFunctionEntryLabel();
}

// Force static initialization.
extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeRISCVAsmPrinter() {
  RegisterAsmPrinter<RISCVAsmPrinter> X(getTheRISCV32Target());
  RegisterAsmPrinter<RISCVAsmPrinter> Y(getTheRISCV64Target());
}

void RISCVAsmPrinter::LowerHWASAN_CHECK_MEMACCESS(const MachineInstr &MI) {
  Register Reg = MI.getOperand(0).getReg();
  uint32_t AccessInfo = MI.getOperand(1).getImm();
  MCSymbol *&Sym =
      HwasanMemaccessSymbols[HwasanMemaccessTuple(Reg, AccessInfo)];
  if (!Sym) {
    // FIXME: Make this work on non-ELF.
    if (!TM.getTargetTriple().isOSBinFormatELF())
      report_fatal_error("llvm.hwasan.check.memaccess only supported on ELF");

    std::string SymName = "__hwasan_check_x" + utostr(Reg - RISCV::X0) + "_" +
                          utostr(AccessInfo) + "_short";
    Sym = OutContext.getOrCreateSymbol(SymName);
  }
  auto Res = MCSymbolRefExpr::create(Sym, OutContext);
  auto Expr = MCSpecifierExpr::create(Res, ELF::R_RISCV_CALL_PLT, OutContext);

  EmitToStreamer(*OutStreamer, MCInstBuilder(RISCV::PseudoCALL).addExpr(Expr));
}

void RISCVAsmPrinter::LowerKCFI_CHECK(const MachineInstr &MI) {
  Register AddrReg = MI.getOperand(0).getReg();
  assert(std::next(MI.getIterator())->isCall() &&
         "KCFI_CHECK not followed by a call instruction");
  assert(std::next(MI.getIterator())->getOperand(0).getReg() == AddrReg &&
         "KCFI_CHECK call target doesn't match call operand");

  // Temporary registers for comparing the hashes. If a register is used
  // for the call target, or reserved by the user, we can clobber another
  // temporary register as the check is immediately followed by the
  // call. The check defaults to X6/X7, but can fall back to X28-X31 if
  // needed.
  unsigned ScratchRegs[] = {RISCV::X6, RISCV::X7};
  unsigned NextReg = RISCV::X28;
  auto isRegAvailable = [&](unsigned Reg) {
    return Reg != AddrReg && !STI->isRegisterReservedByUser(Reg);
  };
  for (auto &Reg : ScratchRegs) {
    if (isRegAvailable(Reg))
      continue;
    while (!isRegAvailable(NextReg))
      ++NextReg;
    Reg = NextReg++;
    if (Reg > RISCV::X31)
      report_fatal_error("Unable to find scratch registers for KCFI_CHECK");
  }

  if (AddrReg == RISCV::X0) {
    // Checking X0 makes no sense. Instead of emitting a load, zero
    // ScratchRegs[0].
    EmitToStreamer(*OutStreamer, MCInstBuilder(RISCV::ADDI)
                                     .addReg(ScratchRegs[0])
                                     .addReg(RISCV::X0)
                                     .addImm(0));
  } else {
    // Adjust the offset for patchable-function-prefix. This assumes that
    // patchable-function-prefix is the same for all functions.
    int NopSize = STI->hasStdExtZca() ? 2 : 4;
    int64_t PrefixNops = 0;
    (void)MI.getMF()
        ->getFunction()
        .getFnAttribute("patchable-function-prefix")
        .getValueAsString()
        .getAsInteger(10, PrefixNops);

    // Load the target function type hash.
    EmitToStreamer(*OutStreamer, MCInstBuilder(RISCV::LW)
                                     .addReg(ScratchRegs[0])
                                     .addReg(AddrReg)
                                     .addImm(-(PrefixNops * NopSize + 4)));
  }

  // Load the expected 32-bit type hash.
  const int64_t Type = MI.getOperand(1).getImm();
  const int64_t Hi20 = ((Type + 0x800) >> 12) & 0xFFFFF;
  const int64_t Lo12 = SignExtend64<12>(Type);
  if (Hi20) {
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(RISCV::LUI).addReg(ScratchRegs[1]).addImm(Hi20));
  }
  if (Lo12 || Hi20 == 0) {
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder((STI->hasFeature(RISCV::Feature64Bit) && Hi20)
                                     ? RISCV::ADDIW
                                     : RISCV::ADDI)
                       .addReg(ScratchRegs[1])
                       .addReg(ScratchRegs[1])
                       .addImm(Lo12));
  }

  // Compare the hashes and trap if there's a mismatch.
  MCSymbol *Pass = OutContext.createTempSymbol();
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(RISCV::BEQ)
                     .addReg(ScratchRegs[0])
                     .addReg(ScratchRegs[1])
                     .addExpr(MCSymbolRefExpr::create(Pass, OutContext)));

  MCSymbol *Trap = OutContext.createTempSymbol();
  OutStreamer->emitLabel(Trap);
  EmitToStreamer(*OutStreamer, MCInstBuilder(RISCV::EBREAK));
  emitKCFITrapEntry(*MI.getMF(), Trap);
  OutStreamer->emitLabel(Pass);
}

void RISCVAsmPrinter::EmitHwasanMemaccessSymbols(Module &M) {
  if (HwasanMemaccessSymbols.empty())
    return;

  assert(TM.getTargetTriple().isOSBinFormatELF());
  // Use MCSubtargetInfo from TargetMachine. Individual functions may have
  // attributes that differ from other functions in the module and we have no
  // way to know which function is correct.
  const MCSubtargetInfo &MCSTI = *TM.getMCSubtargetInfo();

  MCSymbol *HwasanTagMismatchV2Sym =
      OutContext.getOrCreateSymbol("__hwasan_tag_mismatch_v2");
  // Annotate symbol as one having incompatible calling convention, so
  // run-time linkers can instead eagerly bind this function.
  auto &RTS =
      static_cast<RISCVTargetStreamer &>(*OutStreamer->getTargetStreamer());
  RTS.emitDirectiveVariantCC(*HwasanTagMismatchV2Sym);

  const MCSymbolRefExpr *HwasanTagMismatchV2Ref =
      MCSymbolRefExpr::create(HwasanTagMismatchV2Sym, OutContext);
  auto Expr = MCSpecifierExpr::create(HwasanTagMismatchV2Ref,
                                      ELF::R_RISCV_CALL_PLT, OutContext);

  for (auto &P : HwasanMemaccessSymbols) {
    unsigned Reg = std::get<0>(P.first);
    uint32_t AccessInfo = std::get<1>(P.first);
    MCSymbol *Sym = P.second;

    unsigned Size =
        1 << ((AccessInfo >> HWASanAccessInfo::AccessSizeShift) & 0xf);
    OutStreamer->switchSection(OutContext.getELFSection(
        ".text.hot", ELF::SHT_PROGBITS,
        ELF::SHF_EXECINSTR | ELF::SHF_ALLOC | ELF::SHF_GROUP, 0, Sym->getName(),
        /*IsComdat=*/true));

    OutStreamer->emitSymbolAttribute(Sym, MCSA_ELF_TypeFunction);
    OutStreamer->emitSymbolAttribute(Sym, MCSA_Weak);
    OutStreamer->emitSymbolAttribute(Sym, MCSA_Hidden);
    OutStreamer->emitLabel(Sym);

    // Extract shadow offset from ptr
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(RISCV::SLLI).addReg(RISCV::X6).addReg(Reg).addImm(8),
        MCSTI);
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(RISCV::SRLI)
                       .addReg(RISCV::X6)
                       .addReg(RISCV::X6)
                       .addImm(12),
                   MCSTI);
    // load shadow tag in X6, X5 contains shadow base
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(RISCV::ADD)
                       .addReg(RISCV::X6)
                       .addReg(RISCV::X5)
                       .addReg(RISCV::X6),
                   MCSTI);
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(RISCV::LBU).addReg(RISCV::X6).addReg(RISCV::X6).addImm(0),
        MCSTI);
    // Extract tag from pointer and compare it with loaded tag from shadow
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(RISCV::SRLI).addReg(RISCV::X7).addReg(Reg).addImm(56),
        MCSTI);
    MCSymbol *HandleMismatchOrPartialSym = OutContext.createTempSymbol();
    // X7 contains tag from the pointer, while X6 contains tag from memory
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(RISCV::BNE)
                       .addReg(RISCV::X7)
                       .addReg(RISCV::X6)
                       .addExpr(MCSymbolRefExpr::create(
                           HandleMismatchOrPartialSym, OutContext)),
                   MCSTI);
    MCSymbol *ReturnSym = OutContext.createTempSymbol();
    OutStreamer->emitLabel(ReturnSym);
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(RISCV::JALR)
                       .addReg(RISCV::X0)
                       .addReg(RISCV::X1)
                       .addImm(0),
                   MCSTI);
    OutStreamer->emitLabel(HandleMismatchOrPartialSym);

    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(RISCV::ADDI)
                       .addReg(RISCV::X28)
                       .addReg(RISCV::X0)
                       .addImm(16),
                   MCSTI);
    MCSymbol *HandleMismatchSym = OutContext.createTempSymbol();
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(RISCV::BGEU)
            .addReg(RISCV::X6)
            .addReg(RISCV::X28)
            .addExpr(MCSymbolRefExpr::create(HandleMismatchSym, OutContext)),
        MCSTI);

    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(RISCV::ANDI).addReg(RISCV::X28).addReg(Reg).addImm(0xF),
        MCSTI);

    if (Size != 1)
      EmitToStreamer(*OutStreamer,
                     MCInstBuilder(RISCV::ADDI)
                         .addReg(RISCV::X28)
                         .addReg(RISCV::X28)
                         .addImm(Size - 1),
                     MCSTI);
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(RISCV::BGE)
            .addReg(RISCV::X28)
            .addReg(RISCV::X6)
            .addExpr(MCSymbolRefExpr::create(HandleMismatchSym, OutContext)),
        MCSTI);

    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(RISCV::ORI).addReg(RISCV::X6).addReg(Reg).addImm(0xF),
        MCSTI);
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(RISCV::LBU).addReg(RISCV::X6).addReg(RISCV::X6).addImm(0),
        MCSTI);
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(RISCV::BEQ)
                       .addReg(RISCV::X6)
                       .addReg(RISCV::X7)
                       .addExpr(MCSymbolRefExpr::create(ReturnSym, OutContext)),
                   MCSTI);

    OutStreamer->emitLabel(HandleMismatchSym);

    // | Previous stack frames...        |
    // +=================================+ <-- [SP + 256]
    // |              ...                |
    // |                                 |
    // | Stack frame space for x12 - x31.|
    // |                                 |
    // |              ...                |
    // +---------------------------------+ <-- [SP + 96]
    // | Saved x11(arg1), as             |
    // | __hwasan_check_* clobbers it.   |
    // +---------------------------------+ <-- [SP + 88]
    // | Saved x10(arg0), as             |
    // | __hwasan_check_* clobbers it.   |
    // +---------------------------------+ <-- [SP + 80]
    // |                                 |
    // | Stack frame space for x9.       |
    // +---------------------------------+ <-- [SP + 72]
    // |                                 |
    // | Saved x8(fp), as                |
    // | __hwasan_check_* clobbers it.   |
    // +---------------------------------+ <-- [SP + 64]
    // |              ...                |
    // |                                 |
    // | Stack frame space for x2 - x7.  |
    // |                                 |
    // |              ...                |
    // +---------------------------------+ <-- [SP + 16]
    // | Return address (x1) for caller  |
    // | of __hwasan_check_*.            |
    // +---------------------------------+ <-- [SP + 8]
    // | Reserved place for x0, possibly |
    // | junk, since we don't save it.   |
    // +---------------------------------+ <-- [x2 / SP]

    // Adjust sp
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(RISCV::ADDI)
                       .addReg(RISCV::X2)
                       .addReg(RISCV::X2)
                       .addImm(-256),
                   MCSTI);

    // store x10(arg0) by new sp
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(RISCV::SD)
                       .addReg(RISCV::X10)
                       .addReg(RISCV::X2)
                       .addImm(8 * 10),
                   MCSTI);
    // store x11(arg1) by new sp
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(RISCV::SD)
                       .addReg(RISCV::X11)
                       .addReg(RISCV::X2)
                       .addImm(8 * 11),
                   MCSTI);

    // store x8(fp) by new sp
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(RISCV::SD).addReg(RISCV::X8).addReg(RISCV::X2).addImm(8 *
                                                                            8),
        MCSTI);
    // store x1(ra) by new sp
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(RISCV::SD).addReg(RISCV::X1).addReg(RISCV::X2).addImm(1 *
                                                                            8),
        MCSTI);
    if (Reg != RISCV::X10)
      EmitToStreamer(
          *OutStreamer,
          MCInstBuilder(RISCV::ADDI).addReg(RISCV::X10).addReg(Reg).addImm(0),
          MCSTI);
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(RISCV::ADDI)
                       .addReg(RISCV::X11)
                       .addReg(RISCV::X0)
                       .addImm(AccessInfo & HWASanAccessInfo::RuntimeMask),
                   MCSTI);

    EmitToStreamer(*OutStreamer, MCInstBuilder(RISCV::PseudoCALL).addExpr(Expr),
                   MCSTI);
  }
}

void RISCVAsmPrinter::emitNoteGnuProperty(const Module &M) {
  if (const Metadata *const Flag = M.getModuleFlag("cf-protection-return");
      Flag && !mdconst::extract<ConstantInt>(Flag)->isZero()) {
    RISCVTargetStreamer &RTS =
        static_cast<RISCVTargetStreamer &>(*OutStreamer->getTargetStreamer());
    RTS.emitNoteGnuPropertySection(ELF::GNU_PROPERTY_RISCV_FEATURE_1_CFI_SS);
  }
}

static MCOperand lowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym,
                                    const AsmPrinter &AP) {
  MCContext &Ctx = AP.OutContext;
  RISCV::Specifier Kind;

  switch (MO.getTargetFlags()) {
  default:
    llvm_unreachable("Unknown target flag on GV operand");
  case RISCVII::MO_None:
    Kind = RISCV::S_None;
    break;
  case RISCVII::MO_CALL:
    Kind = ELF::R_RISCV_CALL_PLT;
    break;
  case RISCVII::MO_LO:
    Kind = RISCV::S_LO;
    break;
  case RISCVII::MO_HI:
    Kind = ELF::R_RISCV_HI20;
    break;
  case RISCVII::MO_PCREL_LO:
    Kind = RISCV::S_PCREL_LO;
    break;
  case RISCVII::MO_PCREL_HI:
    Kind = ELF::R_RISCV_PCREL_HI20;
    break;
  case RISCVII::MO_GOT_HI:
    Kind = ELF::R_RISCV_GOT_HI20;
    break;
  case RISCVII::MO_TPREL_LO:
    Kind = RISCV::S_TPREL_LO;
    break;
  case RISCVII::MO_TPREL_HI:
    Kind = ELF::R_RISCV_TPREL_HI20;
    break;
  case RISCVII::MO_TPREL_ADD:
    Kind = ELF::R_RISCV_TPREL_ADD;
    break;
  case RISCVII::MO_TLS_GOT_HI:
    Kind = ELF::R_RISCV_TLS_GOT_HI20;
    break;
  case RISCVII::MO_TLS_GD_HI:
    Kind = ELF::R_RISCV_TLS_GD_HI20;
    break;
  case RISCVII::MO_TLSDESC_HI:
    Kind = ELF::R_RISCV_TLSDESC_HI20;
    break;
  case RISCVII::MO_TLSDESC_LOAD_LO:
    Kind = ELF::R_RISCV_TLSDESC_LOAD_LO12;
    break;
  case RISCVII::MO_TLSDESC_ADD_LO:
    Kind = ELF::R_RISCV_TLSDESC_ADD_LO12;
    break;
  case RISCVII::MO_TLSDESC_CALL:
    Kind = ELF::R_RISCV_TLSDESC_CALL;
    break;
  }

  const MCExpr *ME = MCSymbolRefExpr::create(Sym, Ctx);

  if (!MO.isJTI() && !MO.isMBB() && MO.getOffset())
    ME = MCBinaryExpr::createAdd(
        ME, MCConstantExpr::create(MO.getOffset(), Ctx), Ctx);

  if (Kind != RISCV::S_None)
    ME = MCSpecifierExpr::create(ME, Kind, Ctx);
  return MCOperand::createExpr(ME);
}

bool RISCVAsmPrinter::lowerOperand(const MachineOperand &MO,
                                   MCOperand &MCOp) const {
  switch (MO.getType()) {
  default:
    report_fatal_error("lowerOperand: unknown operand type");
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit())
      return false;
    MCOp = MCOperand::createReg(MO.getReg());
    break;
  case MachineOperand::MO_RegisterMask:
    // Regmasks are like implicit defs.
    return false;
  case MachineOperand::MO_Immediate:
    MCOp = MCOperand::createImm(MO.getImm());
    break;
  case MachineOperand::MO_MachineBasicBlock:
    MCOp = lowerSymbolOperand(MO, MO.getMBB()->getSymbol(), *this);
    break;
  case MachineOperand::MO_GlobalAddress:
    MCOp = lowerSymbolOperand(MO, getSymbolPreferLocal(*MO.getGlobal()), *this);
    break;
  case MachineOperand::MO_BlockAddress:
    MCOp = lowerSymbolOperand(MO, GetBlockAddressSymbol(MO.getBlockAddress()),
                              *this);
    break;
  case MachineOperand::MO_ExternalSymbol:
    MCOp = lowerSymbolOperand(MO, GetExternalSymbolSymbol(MO.getSymbolName()),
                              *this);
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    MCOp = lowerSymbolOperand(MO, GetCPISymbol(MO.getIndex()), *this);
    break;
  case MachineOperand::MO_JumpTableIndex:
    MCOp = lowerSymbolOperand(MO, GetJTISymbol(MO.getIndex()), *this);
    break;
  case MachineOperand::MO_MCSymbol:
    MCOp = lowerSymbolOperand(MO, MO.getMCSymbol(), *this);
    break;
  }
  return true;
}

static bool lowerRISCVVMachineInstrToMCInst(const MachineInstr *MI,
                                            MCInst &OutMI,
                                            const RISCVSubtarget *STI) {
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(MI->getOpcode());
  if (!RVV)
    return false;

  OutMI.setOpcode(RVV->BaseInstr);

  const TargetInstrInfo *TII = STI->getInstrInfo();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  assert(TRI && "TargetRegisterInfo expected");

  const MCInstrDesc &MCID = MI->getDesc();
  uint64_t TSFlags = MCID.TSFlags;
  unsigned NumOps = MI->getNumExplicitOperands();

  // Skip policy, SEW, VL, VXRM/FRM operands which are the last operands if
  // present.
  if (RISCVII::hasVecPolicyOp(TSFlags))
    --NumOps;
  if (RISCVII::hasSEWOp(TSFlags))
    --NumOps;
  if (RISCVII::hasVLOp(TSFlags))
    --NumOps;
  if (RISCVII::hasRoundModeOp(TSFlags))
    --NumOps;

  bool hasVLOutput = RISCVInstrInfo::isFaultOnlyFirstLoad(*MI);
  for (unsigned OpNo = 0; OpNo != NumOps; ++OpNo) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    // Skip vl output. It should be the second output.
    if (hasVLOutput && OpNo == 1)
      continue;

    // Skip passthru op. It should be the first operand after the defs.
    if (OpNo == MI->getNumExplicitDefs() && MO.isReg() && MO.isTied()) {
      assert(MCID.getOperandConstraint(OpNo, MCOI::TIED_TO) == 0 &&
             "Expected tied to first def.");
      const MCInstrDesc &OutMCID = TII->get(OutMI.getOpcode());
      // Skip if the next operand in OutMI is not supposed to be tied. Unless it
      // is a _TIED instruction.
      if (OutMCID.getOperandConstraint(OutMI.getNumOperands(), MCOI::TIED_TO) <
              0 &&
          !RISCVII::isTiedPseudo(TSFlags))
        continue;
    }

    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      llvm_unreachable("Unknown operand type");
    case MachineOperand::MO_Register: {
      Register Reg = MO.getReg();

      if (RISCV::VRM2RegClass.contains(Reg) ||
          RISCV::VRM4RegClass.contains(Reg) ||
          RISCV::VRM8RegClass.contains(Reg)) {
        Reg = TRI->getSubReg(Reg, RISCV::sub_vrm1_0);
        assert(Reg && "Subregister does not exist");
      } else if (RISCV::FPR16RegClass.contains(Reg)) {
        Reg =
            TRI->getMatchingSuperReg(Reg, RISCV::sub_16, &RISCV::FPR32RegClass);
        assert(Reg && "Subregister does not exist");
      } else if (RISCV::FPR64RegClass.contains(Reg)) {
        Reg = TRI->getSubReg(Reg, RISCV::sub_32);
        assert(Reg && "Superregister does not exist");
      } else if (RISCV::VRN2M1RegClass.contains(Reg) ||
                 RISCV::VRN2M2RegClass.contains(Reg) ||
                 RISCV::VRN2M4RegClass.contains(Reg) ||
                 RISCV::VRN3M1RegClass.contains(Reg) ||
                 RISCV::VRN3M2RegClass.contains(Reg) ||
                 RISCV::VRN4M1RegClass.contains(Reg) ||
                 RISCV::VRN4M2RegClass.contains(Reg) ||
                 RISCV::VRN5M1RegClass.contains(Reg) ||
                 RISCV::VRN6M1RegClass.contains(Reg) ||
                 RISCV::VRN7M1RegClass.contains(Reg) ||
                 RISCV::VRN8M1RegClass.contains(Reg)) {
        Reg = TRI->getSubReg(Reg, RISCV::sub_vrm1_0);
        assert(Reg && "Subregister does not exist");
      }

      MCOp = MCOperand::createReg(Reg);
      break;
    }
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::createImm(MO.getImm());
      break;
    }
    OutMI.addOperand(MCOp);
  }

  // Unmasked pseudo instructions need to append dummy mask operand to
  // V instructions. All V instructions are modeled as the masked version.
  const MCInstrDesc &OutMCID = TII->get(OutMI.getOpcode());
  if (OutMI.getNumOperands() < OutMCID.getNumOperands()) {
    assert(OutMCID.operands()[OutMI.getNumOperands()].RegClass ==
               RISCV::VMV0RegClassID &&
           "Expected only mask operand to be missing");
    OutMI.addOperand(MCOperand::createReg(RISCV::NoRegister));
  }

  assert(OutMI.getNumOperands() == OutMCID.getNumOperands());
  return true;
}

bool RISCVAsmPrinter::lowerToMCInst(const MachineInstr *MI, MCInst &OutMI) {
  if (lowerRISCVVMachineInstrToMCInst(MI, OutMI, STI))
    return false;

  OutMI.setOpcode(MI->getOpcode());

  for (const MachineOperand &MO : MI->operands()) {
    MCOperand MCOp;
    if (lowerOperand(MO, MCOp))
      OutMI.addOperand(MCOp);
  }

  switch (OutMI.getOpcode()) {
  case TargetOpcode::PATCHABLE_FUNCTION_ENTER: {
    const Function &F = MI->getParent()->getParent()->getFunction();
    if (F.hasFnAttribute("patchable-function-entry")) {
      unsigned Num;
      if (F.getFnAttribute("patchable-function-entry")
              .getValueAsString()
              .getAsInteger(10, Num))
        return false;
      emitNops(Num);
      return true;
    }
    break;
  }
  }
  return false;
}

void RISCVAsmPrinter::emitMachineConstantPoolValue(
    MachineConstantPoolValue *MCPV) {
  auto *RCPV = static_cast<RISCVConstantPoolValue *>(MCPV);
  MCSymbol *MCSym;

  if (RCPV->isGlobalValue()) {
    auto *GV = RCPV->getGlobalValue();
    MCSym = getSymbol(GV);
  } else {
    assert(RCPV->isExtSymbol() && "unrecognized constant pool type");
    auto Sym = RCPV->getSymbol();
    MCSym = GetExternalSymbolSymbol(Sym);
  }

  const MCExpr *Expr = MCSymbolRefExpr::create(MCSym, OutContext);
  uint64_t Size = getDataLayout().getTypeAllocSize(RCPV->getType());
  OutStreamer->emitValue(Expr, Size);
}

char RISCVAsmPrinter::ID = 0;

INITIALIZE_PASS(RISCVAsmPrinter, "riscv-asm-printer", "RISC-V Assembly Printer",
                false, false)
