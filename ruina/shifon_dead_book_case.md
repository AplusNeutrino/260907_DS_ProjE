# Shifon × 死者之書 — Event Investigation

Status: active investigation

## User-observed state
- User currently possesses 死者之書 上 / 中 / 下.
- Shifon does **not** have the three corresponding skills:
  - 吸血鬼の呪文 / 吸血鬼的咒文
  - 死の空気 / 死之空气
  - 瘴気 / 瘴气
- The exchange/request event has stopped appearing despite repeated library searches.
- User previously triggered/gave a Dead Book to Shifon, then reloaded an earlier save, and suspects a state-related issue.
- User reports having rested with Shifon many times (possibly 15+), raising a separate hypothesis about high affection.

## Conclusions carried forward from prior chat
These are provisional until re-verified against authoritative event data/code:
- No evidence yet that reloading after an exchange creates a cross-save persistent completion flag.
- High Shifon affection is not currently considered a likely blocker for this event.
- The absence of all three corresponding skills suggests the normal 'already learned' gating condition should not be satisfied.
- The highest-value next step is to inspect the exact event conditions for `シーとアイテム争奪` and all switches/variables it branches on.

## Current save baseline
- Filename: `Save01.lsd`
- SHA-256: `ab849fc46fbc4818a1da150bd44e61d67b9df8c3d84708230da671a60b5605aa`
- Size: ~37 KB
- Treat this as read-only unless the user explicitly asks for a modified save.

## Previously reported save observations
These came from the earlier conversation and should be independently reproduced where possible:
- `宿にシーフォンがいる` = ON
- `シーフォン去る` = OFF
- Party reportedly contains protagonist + Fran + Shifon
- `図書館死者の書中入手` = ON
- No obvious indication that `シーとアイテム争奪` was left in an abnormal in-progress state

## Hypotheses to test
1. Event condition/branch not satisfied despite visible inventory/skills.
2. Hidden variable or switch related to library-search completion or Shifon item contest state.
3. Localized/Chinese build changed event logic or IDs.
4. Save serialization/state discrepancy (visible state differs from event-condition state).
5. RNG/process misunderstanding rather than corruption.

## Investigation plan
- Identify exact Ruina 1.21 event definition for `シーとアイテム争奪`.
- Enumerate every prerequisite, switch, variable, inventory check, party/member check, and learned-skill check.
- Parse or inspect `Save01.lsd` for each relevant field.
- Compare the save state against the event predicate step by step.
- Search Japanese/Chinese discussions for the exact non-trigger symptom and verified fixes.
- Keep spoilers outside this event out of the notes unless strictly necessary.

## Evidence log
Add dated entries below as findings are reproduced.

### 2026-09-10
- Workspace created.
- Current `Save01.lsd` fingerprint recorded.
- Detailed save/event verification still pending in this continuation chat.
