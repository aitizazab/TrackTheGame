"""The pre-D26 prompt and schemas, frozen.

Restored verbatim from git 2bd4d18^ - the commit before "Rewrite prompt and
schemas around judgement calls". This module exists for ONE reason: the D26
rewrite has never been run, and comparing it against what it replaced requires
what it replaced to still be runnable. Every measurement in this project before
3 Sep, including the signed-off deliverable video, came from the definitions in
this file.

DO NOT EDIT to "improve" anything. This is a control, and its whole value is
being byte-identical to what produced those numbers. If a bug is found here, it
was in the shipped runs too - that is a finding, not something to patch.

What differs from the live definitions in detect.py, which is exactly what the
prompt arm of the experiment measures:

  - PROMPT_V1 is 3291 characters against 440. Every rule is stated twice, once
    here as prose and once in a schema description.
  - players carry a model-reported "conf"; COMPACT_ORDER_V1 is 8 fields, not 7.
  - the "kits" / "accent" block exists, at the top level of every frame.
  - the eleven ball decoys are listed by name, inside a negation.
  - the worked coordinate example, and the retracted "the marker floats below
    their feet" clause, are both present.
"""

SCHEMA_V1 = {
    "name": "frame_detections",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["scene", "kits", "players", "ball"],
        "properties": {
            "scene": {
                "type": "string",
                "description": "One sentence: camera framing, lighting, and the "
                               "two kit colours. Written BEFORE looking for "
                               "positions, as working-out."
            },
            "kits": {
                "type": "array",
                "description": "The distinct outfield kits visible, most common "
                               "first. Usually exactly two.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["colour", "accent"],
                    "properties": {
                        "colour": {"type": "string",
                                   "description": "dominant shirt colour, one "
                                                  "common word"},
                        # The renderer needs a second colour to fall back on when
                        # the two kits are too close to tell apart at a glance.
                        # Asked once per frame rather than once per player: it is
                        # a property of the kit, and per-player would cost ~20
                        # extra output tokens per person for no extra signal.
                        "accent": {"type": ["string", "null"],
                                   "description": "secondary colour on that kit "
                                                  "— trim, sleeves, shorts, or "
                                                  "the number itself. null if "
                                                  "the kit is plain."}
                    }
                }
            },
            "players": {
                "type": "array",
                "description": "One entry per player on the field of play, "
                               "GOALKEEPERS INCLUDED. Empty array is valid and "
                               "correct if none are visible.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x", "y", "w", "h", "kit", "num", "role", "conf"],
                    "properties": {
                        "x": {"type": "number",
                              "description": "LEFT edge of the player's box, "
                                             "fraction of image width, 0.0-1.0"},
                        "y": {"type": "number",
                              "description": "TOP edge of the player's box, "
                                             "fraction of image height, 0.0-1.0"},
                        "w": {"type": "number",
                              "description": "box width, fraction of image width"},
                        "h": {"type": "number",
                              "description": "box height, fraction of image height"},
                        "kit": {"type": "string",
                                "description": "Shirt colour as one common word"},
                        "num": {"type": ["integer", "null"],
                                "description": "Jersey number ONLY if you can "
                                               "actually read it. null otherwise."},
                        # Without this a goalkeeper in a third kit colour is
                        # indistinguishable from an unstable colour word, and the
                        # renderer cannot tell "green kit = keeper" from "someone
                        # said green once by mistake".
                        "role": {"type": "string", "enum": ["outfield", "goalkeeper"],
                                 "description": "goalkeeper if they wear a "
                                                "different kit from both teams "
                                                "and stand in/near a goal. In "
                                                "sports with no goalkeeper, "
                                                "every player is outfield"},
                        "conf": {"type": "number",
                                 "description": "0.0 to 1.0, how sure you are this "
                                                "is a player at this position"}
                    }
                }
            },
            "ball": {
                "type": ["object", "null"],
                "description": "null when the ball is not visible. It often is not.",
                "additionalProperties": False,
                # TRIED AND REMOVED 3 Sep: a `kind` field naming which sport's
                # ball this is, so the clip's sport could be set by majority vote
                # and minority reports rejected as misidentifications. On
                # basketball all 136 detections said "basketball" — including
                # every one the geometric filters rejected — so the wrong-sport
                # filter never fired. The model names the sport it is watching,
                # not the object: by the time it fills the field it has already
                # decided "this is the ball", so the field sits downstream of the
                # error rather than checking it. Cost 6.8% in tokens for nothing.
                #
                # The clothing line in the prompt, added at the same time, DID
                # work — see the note there.
                "required": ["x", "y", "w", "h", "conf"],
                "properties": {
                    "x": {"type": "number", "description": "left edge, fraction"},
                    "y": {"type": "number", "description": "top edge, fraction"},
                    "w": {"type": "number", "description": "width, fraction"},
                    "h": {"type": "number", "description": "height, fraction"},
                    "conf": {"type": "number"}
                }
            }
        }
    }
}


PROMPT_V1 = """You are looking at one frame of sports footage.

Report the PLAYERS and the BALL.

Give each one a BOUNDING BOX in fractions of the image, never in pixels.
  x = left edge of the box    (0.0 = image left,  1.0 = image right)
  y = top edge of the box     (0.0 = image top,   1.0 = image bottom)
  w = box width               (as a fraction of the image width)
  h = box height              (as a fraction of the image height)

  Worked example. A player standing in the middle of the picture, occupying the
  lower half vertically and a narrow slice horizontally:
      x = 0.48, y = 0.50, w = 0.04, h = 0.28
  Their box therefore spans 0.48-0.52 across and 0.50-0.78 down.

  The box must be TIGHT: top edge at the top of their head, bottom edge where
  their feet meet the ground. The bottom edge is used to place a marker under
  them, so if the box runs long the marker floats below their feet.

For each player also give:
  kit  the colour of their SHIRT, as one ordinary word: red, blue, white,
       yellow, green, black, orange, purple. Judge the colour itself. Do not
       call them "team A" or "home"; another frame will be judged separately
       and the colours must agree between them.
  num  the number on their shirt ONLY IF YOU CAN GENUINELY READ IT. If their
       back is turned, if it is blurred, if they are too small, if it is
       covered - use null. A wrong number is far worse than no number.
  conf 1.0 you are certain, 0.5 you think so, 0.2 you are guessing.

Rules that matter:
  - Report players on the field of play - the pitch, court, or playing surface.
  - GOALKEEPERS. If this sport has a goalkeeper (football, hockey, handball,
    futsal), they ARE players and must be reported even though their kit matches
    neither team. Mark them role="goalkeeper". If the sport has no goalkeeper
    (basketball, volleyball), every player is role="outfield". Decide from what
    you can see in the image; do not assume the sport.
  - Do NOT report match officials (referees, umpires, linesmen), substitutes or
    players on the bench, coaches, medical staff, the crowd, or ball boys.
  - Report every player you can see, including partly hidden ones. Give a partly
    hidden player a low conf rather than leaving them out.
  - Do NOT pad the list to a round number. If you can see 7 players, report 7.
    There is no expected count, and it does not depend on the sport.

THE BALL:
  - A tight box, same fraction format.
  - Markings painted on the playing surface are not the ball - centre spots,
    penalty spots, painted arcs, court lines and logos. Check that what you are
    looking at sits ABOVE the surface rather than being printed onto it.
  - Worn or carried objects are not the ball either: a boot, sock, glove,
    shinpad, bandage or a bunched sleeve. Pale and roundish is not enough.
  - If you cannot see the ball, set ball to null. Do not place it where you
    think it ought to be, and do not settle for the nearest small round thing.

In "kits", list the two teams' kits. For each, give the dominant shirt colour and
one secondary "accent" colour — the trim, sleeves, shorts, or the colour the
numbers are printed in. If a kit is genuinely plain, accent is null.

Fill in "scene" first, as working-out, before you give any coordinates."""


COMPACT_ORDER_V1 = ["x", "y", "w", "h", "kit", "num", "role", "conf"]


COMPACT_SCHEMA_V1 = {
    "name": "frame_detections_compact",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["scene", "kits", "players", "ball"],
        "properties": {
            # DESCRIPTIONS RESTORED 3 Sep. The array format and the description
            # cut went in together and were measured together as one -29%, which
            # hid that they pull in opposite directions:
            #
            #   --terse-schema  strips descriptions, keeps objects  -> output +11%
            #   --compact       strips descriptions, uses arrays    -> output -29%
            #
            # Dropping the key names is worth about -40%; stripping the
            # descriptions costs about +11% on top of it, because a model given
            # less guidance reasons for longer. The cut was never a saving, it
            # was a tax the array format was paying. So keep the format and give
            # the guidance back — every semantic line below is carried over
            # verbatim from SCHEMA, which is the version that was measured to
            # earn its tokens.
            "scene": {"type": "string",
                      "description": "One sentence: camera framing, lighting, and "
                                     "the two kit colours. Written BEFORE looking "
                                     "for positions, as working-out."},
            "kits": {"type": "array",
                     "description": "The distinct outfield kits visible, most "
                                    "common first. Usually exactly two.",
                     "items": {
                         "type": "object", "additionalProperties": False,
                         "required": ["colour", "accent"],
                         "properties": {
                             "colour": {"type": "string",
                                        "description": "dominant shirt colour, "
                                                       "one common word"},
                             "accent": {"type": ["string", "null"],
                                        "description": "secondary colour on that "
                                                       "kit - trim, sleeves, "
                                                       "shorts, or the number "
                                                       "itself. null if the kit "
                                                       "is plain."}}}},
            "players": {
                "type": "array",
                # The array form loses per-field typing entirely: `items` has to
                # admit number, string and null, so nothing stops position 0
                # being a string or position 5 a float. In the object form `x`
                # was constrained to number and `num` to integer|null. This
                # sentence is now the ONLY thing carrying that contract, which is
                # why it states the type of every position as well as its meaning.
                "description": ("One array per player, GOALKEEPERS INCLUDED, "
                                "ALWAYS in this order: "
                                "[x, y, w, h, kit, num, role, conf]. "
                                "x = LEFT edge of the box, fraction of image "
                                "width, 0.0-1.0. "
                                "y = TOP edge of the box, fraction of image "
                                "height, 0.0-1.0. "
                                "w = box width as a fraction of image width. "
                                "h = box height as a fraction of image height. "
                                "kit = shirt colour as one common word. "
                                "num = jersey number as an integer ONLY if you "
                                "can actually read it, otherwise null. "
                                "role = \"goalkeeper\" if they wear a different "
                                "kit from both teams and stand in/near a goal, "
                                "otherwise \"outfield\"; in sports with no "
                                "goalkeeper every player is \"outfield\". "
                                "conf = 0.0 to 1.0, how sure you are this is a "
                                "player at this position. "
                                "An empty array is valid and correct if no "
                                "players are visible."),
                "items": {"type": "array",
                          "items": {"type": ["number", "string", "null"]}}},
            "ball": {"type": ["array", "null"],
                     "description": ("[x, y, w, h, conf] or null. null when the "
                                     "ball is not visible, which it often is not. "
                                     "x = left edge, y = top edge, w = width, "
                                     "h = height, all as fractions 0.0-1.0. "
                                     "conf = 0.0 to 1.0."),
                     "items": {"type": ["number", "null"]}},
        }}}
