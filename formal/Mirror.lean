/-
`lake exe mirror` writes rational witnesses for the stencil map.

The values are dyadic, or integer Pythagorean data whose norm is an
integer, so a binary64 division of the reduced fraction matches the
Rust evaluation bit for bit. Goswami, Gunde, Jónsson, Front. Chem. (2026),
doi:10.3389/fchem.2026.1807063, Eq. 9.
-/

def dot (v f : List Rat) : Rat :=
  (v.zip f).foldl (fun acc p => acc + p.1 * p.2) 0

def reflectRat (v f : List Rat) : List Rat :=
  let d := dot v f
  (f.zip v).map fun p => p.1 - (2 : Rat) * d * p.2

def normSq (v : List Rat) : Rat := dot v v

/-- Energy-weighted spring. When `eMax = eRef` the quotient is undefined and
the continuous extension on `eI ≤ eMax` is `kMin`. -/
def energyWeighted (eI eMax eRef kMin kMax : Rat) : Rat :=
  if eMax == eRef then kMin
  else if eRef < eI then
    let α := min 1 (max 0 ((eMax - eI) / (eMax - eRef)))
    (1 - α) * kMax + α * kMin
  else kMin

def ratJson (r : Rat) : String :=
  s!"\"{r.num}/{r.den}\""

def listJson (xs : List Rat) : String :=
  "[" ++ ", ".intercalate (xs.map ratJson) ++ "]"

def reflectCase (v f : List Rat) : String :=
  let out := reflectRat v f
  let zeros := List.replicate v.length (0 : Rat)
  "{\"v\": " ++ listJson v ++ ", \"f\": " ++ listJson f ++ ", \"dneb\": " ++ listJson zeros ++
    ", \"out\": " ++ listJson out ++ "}"

-- Normalise by an integer norm supplied by the caller.
def scaleByNorm (v : List Rat) (n : Rat) : List Rat :=
  v.map fun x => x / n

def tangentCase (next prev : List Rat) (energy energyPrev energyNext norm : Rat) : String :=
  let out := scaleByNorm next norm
  "{\"kind\": \"improved\", \"next\": " ++ listJson next ++ ", \"prev\": " ++ listJson prev ++
    ", \"energy\": " ++ ratJson energy ++ ", \"energy_prev\": " ++ ratJson energyPrev ++
    ", \"energy_next\": " ++ ratJson energyNext ++ ", \"norm\": " ++ ratJson norm ++
    ", \"out\": " ++ listJson out ++ "}"

def translatedCase (g mode : List Rat) (curvature : Rat) : String :=
  let parallel := dot g mode
  let out :=
    if curvature < 0 then (g.zip mode).map fun p => p.1 - (2 : Rat) * parallel * p.2
    else mode.map fun m => (-parallel) * m
  "{\"g\": " ++ listJson g ++ ", \"mode\": " ++ listJson mode ++ ", \"curvature\": " ++
    ratJson curvature ++ ", \"out\": " ++ listJson out ++ "}"

def weightedCase (eI eMax eRef kMin kMax : Rat) : String :=
  let k := energyWeighted eI eMax eRef kMin kMax
  "{\"e_i\": " ++ ratJson eI ++ ", \"e_max\": " ++ ratJson eMax ++ ", \"e_ref\": " ++ ratJson eRef ++
    ", \"k_min\": " ++ ratJson kMin ++ ", \"k_max\": " ++ ratJson kMax ++ ", \"k\": " ++ ratJson k ++ "}"

def join (xs : List String) : String := ",\n".intercalate xs

def main : IO Unit := do
  let reflects := [
    reflectCase [1, 0] [(1 : Rat) / 2, -(1 : Rat) / 4],
    reflectCase [0, -1] [2, 4],
    reflectCase [1, 0, 0] [3 / 4, -1 / 2, 1 / 8]
  ]
  -- 3-4-5: next = prev = (3, 4), equal energy drop 2, raw = (12, 16), norm 20,
  -- unit tangent (3/5, 4/5). The caller passes the integer norm of `next`.
  let tangents := [
    tangentCase [3, 4] [3, 4] 2 0 0 5,
    tangentCase [5, 12] [5, 12] 1 0 0 13,
    tangentCase [1, 0] [0, 1] 1 0 2 1
  ]
  let translated := [
    translatedCase [1, 2] [1, 0] (-1),
    translatedCase [(1 : Rat) / 2, -1 / 4] [1, 0] (-2),
    translatedCase [1, 2] [1, 0] 1
  ]
  let weighted := [
    weightedCase (5 / 2) 4 1 1 5,
    weightedCase 4 4 1 1 5,
    weightedCase 1 4 1 1 5,
    weightedCase 0 4 1 1 5,
    weightedCase 3 3 3 2 7
  ]
  let json :=
    "{\n  \"reflect\": [\n" ++ join reflects ++
    "\n  ],\n  \"tangent\": [\n" ++ join tangents ++
    "\n  ],\n  \"translated\": [\n" ++ join translated ++
    "\n  ],\n  \"energy_weighted\": [\n" ++ join weighted ++
    "\n  ]\n}\n"
  IO.FS.createDirAll "golden"
  IO.FS.writeFile "golden/stencil.json" json
  IO.println "wrote golden/stencil.json"

#guard reflectRat [1, 0] [(1 : Rat) / 2, -(1 : Rat) / 4] == [-(1 : Rat) / 2, -(1 : Rat) / 4]
#guard energyWeighted (5 / 2) 4 1 1 5 == 3
#guard energyWeighted 4 4 1 1 5 == 5
#guard energyWeighted 1 4 1 1 5 == 1
#guard energyWeighted 3 3 3 2 7 == 2
#guard scaleByNorm [3, 4] 5 == [(3 : Rat) / 5, 4 / 5]
