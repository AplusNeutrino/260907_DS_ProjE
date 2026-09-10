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

## Current save baseline
- Filename: `Save01.lsd`
- File type: RPG Maker 2000 `LcfSaveData`
- Size: 37,426 bytes
- SHA-256: `ab849fc46fbc4818a1da150bd44e61d67b9df8c3d84708230da671a60b5605aa`
- Treat this as read-only unless the user explicitly asks for a modified save.

## Reproduced save-state evidence
Parsed directly from the current uploaded `Save01.lsd` using the RPG Maker 2000 LCF/LSD field layout.

- Switch 145 `宿にシーフォンがいる` = **ON**
- Switch 618 `シーフォン去る` = **OFF**
- Variable 225 `シーフォン好感度` = **27**
- The save contains 942 switches and 453 variables; the relevant indices are present and structurally readable.
- No conclusion should be drawn from the common-event runtime block alone: a called common event need not remain stored as an active process after completion.

## External event documentation re-verified
Current and archived Japanese strategy documentation agree on the core behavior:

- Possessing any of 死者之書 上 / 中 / 下 is enough for the event to remain eligible.
- The event can occur when leaving a library / book-storage search area.
- It repeats until Shifon has learned the corresponding skill(s).
- Shifon does **not** generally need to be in the active party for this Dead Book event.
- Accepting an exchange adds +3 Shifon affection per exchange.
- No Shifon-affection upper-bound condition is documented for this event. The affection threshold documented for another Shifon event is unrelated.
- A separate skill-book reference describes the library-exit occurrence as a 50% chance.

Therefore Shifon affection = 27 is notable but is **not currently supported as a blocker** for the Dead Book event.

## Original / Chinese data sources discovered
The public `Xenolies/Ruina-Fix` repository contains:

- Original Japanese Ruina ver1.21 game data (`Ruina/JP`).
- Chinese source data (`Ruina/CN_SourceFile`).
- A Chinese fixed build (`Ruina/CN_Fix`).
- Root `CommonEvent.txt`, where common event **98 = `シーとアイテム争奪`**.

Relevant database fingerprints:
- JP `RPG_RT.ldb`: 1,221,218 bytes, Git blob `f069d6836470057badb7c92fd0b537d137703d03`
- CN source `RPG_RT.ldb`: 1,203,111 bytes, Git blob `81c64d574f20658d6c7e5d6e2f6831a6b6dea2f7`
- CN fix `RPG_RT.ldb`: 1,203,960 bytes, Git blob `8cf73b9496caf9b4eead21dca87e4e1ed9b1963c`

The uploaded save contains Chinese/GBK-compatible text, so the Chinese-build branch is especially relevant to compare against the Japanese 1.21 baseline.

## Current assessment
### Strongly downgraded hypotheses
- **Affection too high / maxed**: no evidence this disables the event; current save affection is 27 and the documented Dead Book trigger has no affection gate.
- **Shifon has left permanently**: contradicted by switch state (`宿にシーフォンがいる` ON; `シーフォン去る` OFF).
- **Cross-save persistence after loading**: no mechanism/evidence found so far that an exchange in a discarded save can persist into an earlier loaded save.

### Still open
1. Exact branch conditions inside common event 98 `シーとアイテム争奪`.
2. Exact actor ID for Shifon and exact skill IDs for the three Dead Book skills, so the save can be checked at the raw actor-skill-array level rather than only via the UI report.
3. Exact item IDs/counts for 死者之書 上 / 中 / 下 in the current save.
4. Whether the user's Chinese build corresponds to the source build, fixed build, or another translation, and whether that build changed the event call site or event 98 logic.
5. Whether the observed non-trigger is a real build/save bug or an RNG/trigger-cycle issue.
6. Whether the earlier claim that 'Shifon in active party makes it guaranteed' has a primary/source-level basis. Current event documentation says party presence is not required, so this claim should not be treated as established yet.

## Next verification steps
- Resolve Shifon actor ID and Dead Book item/skill IDs.
- Parse current inventory and Shifon skill vector directly from `Save01.lsd`.
- Recover/compare the exact event 98 commands in JP 1.21 vs Chinese source/fix data.
- Search bug reports, fix notes, commits, archived discussions, and RTA notes for this exact non-trigger symptom.
- If a concrete broken flag/variable is identified, propose a minimal repair and make a separate patched copy only if the user requests it.

## Evidence log
### 2026-09-10
- Workspace created.
- Current `Save01.lsd` fingerprint recorded.
- LSD structure parsed successfully.
- Verified `宿にシーフォンがいる` = ON.
- Verified `シーフォン去る` = OFF.
- Verified `シーフォン好感度` = 27.
- Re-verified from Japanese strategy documentation that affection is not a documented Dead Book-event gate and Shifon need not be in the active party.
- Located original JP 1.21, Chinese source, and Chinese fixed game databases for later exact logic comparison.
