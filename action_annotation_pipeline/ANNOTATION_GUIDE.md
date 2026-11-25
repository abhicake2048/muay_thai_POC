# Muay Thai action annotation guide v2

## Annotation unit

One event is one atomic action performed by one canonical fighter. Two fighters
attacking simultaneously produce two overlapping events. Split combinations
into individual strikes and connect them with `combo_id`.

Anatomical left/right always belongs to the acting fighter. Never label side
from the left or right of the screen. Store lead/rear separately because stance
switches can produce a right-hand jab or a left-hand cross.

## Temporal boundaries

- `start_sec`: first committed action movement; exclude unrelated footwork.
- `peak_sec`: contact, closest approach for a miss, or maximum extension.
- `end_sec`: substantial recovery to guard or transition to the next action.

## Factorized action fields

For every action verify:

1. `action_family` and `technique`;
2. `anatomical_side` and `effector_limb`;
3. `stance_at_start` and `stance_role`;
4. `target_zone`, `outcome`, and `contact_surface`;
5. actor, target, visibility, and boundaries.

`effector_id` is derived from side and limb. For example, a southpaw jab may
be `technique=jab`, `anatomical_side=right`, `effector_id=right_hand`, and
`stance_role=lead`. Do not change it to a cross merely because the right hand
was used.

If evidence is insufficient, use `unknown` or `uncertain`. Do not infer through
heavy occlusion. Punch trajectory/line is intentionally not part of v2.

## Technique taxonomy

- Punches: jab, cross, hook, uppercut, overhand, superman punch, spinning
  backfist, other punch.
- Kicks: roundhouse kick, teep, side kick, question-mark kick, axe kick,
  spinning back kick, other kick.
- Knees: straight, curved, jumping/flying, other.
- Elbows: horizontal, diagonal, upward, downward, spinning, other.
- Clinch: entry, control, break.
- Defenses: cover, block, parry, slip, duck, lean back, step evade, kick
  check/catch, frame, clinch escape, other.
- Balance: knockdown, fall, trip, balance slip, off-balance, sweep/dump.

Boxing6 lead/rear hook and uppercut labels map to the base technique plus the
`stance_role` attribute. Low/body/head kick labels map to `roundhouse_kick`
plus their target zone.

## Outcome and linked events

Use attempted, clean contact, partial contact, blocked, caught, dodged, missed,
or uncertain. The attack and the defender's response are separate events.
Connect counters using `counter_to_event_id`; connect resulting knockdowns or
balance events through `possible_cause` without claiming causality when the
video is ambiguous.

## Auxiliary intervals

Record stance, ring position, movement, combinations, counterattacks, patterns,
and pace changes. They remain in the master data but are not primary v1
PoseC3D/TriDet action classes.

Add a verified `review_range` for every interval watched exhaustively. Empty
time inside it may be sampled as background. Add `ignore` for replays, camera
cuts, unusable footage, or deliberately unreviewed time. Never convert
unreviewed time to background.

## Quality rules

- Every automatic proposal must be watched and verified or rejected.
- Confirm fighter identity before action attributes.
- Keep uncertain observations but exclude them from verified training exports.
- Double-review at least 10 percent of the pilot.
- Keep every source video entirely in one train/validation/test split.
