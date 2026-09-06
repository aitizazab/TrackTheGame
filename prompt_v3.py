"""v3: v2 with the output slimmed and the frame-edge rule made explicit.

Three changes from v2, each with its own justification, all of them affecting
what the model EMITS rather than how it looks at the image.

1. `role` REMOVED.
   Measured 4 Sep: the field appears 0 times in track.py and 0 times in
   render.py. It is emitted, validated, stored in the detections file, and read
   by nothing. The schema comment claiming the renderer needs it to tell
   "green kit = keeper" from a flickered colour word describes something that
   was never built. Worth ~16% of the players array.
   The GOALKEEPER INSTRUCTION STAYS IN THE PROMPT - that is D14, where the
   single word "outfield" excluded every keeper from every frame. What goes is
   the per-player field, not the requirement to report keepers.

2. COORDINATES AS INTEGERS 0-1000 instead of decimal fractions.
   "0.121" tokenises as roughly three tokens; "121" as one. Measured on a real
   20-player frame: coordinate characters fall 384 -> 197, a 49% cut, which is
   ~17% of the whole emitted payload. Resolution is 1/1000 of the frame, or
   1.9px at 1080p - far finer than the ~100px localisation jitter we spent the
   day chasing, so no precision is lost that we could have used.
   NOTE: this makes v3 a THOUSANDTH convention while v1/v2 are FRACTION, and
   the convention is normally pinned per MODEL. It is declared here per PROMPT
   SET and must be verified on the first run - two Gemini lite models already
   return 0-1000 natively, so the space is not exotic, but "the model obeys the
   schema" is exactly the assumption COORD_CONVENTION exists to refuse.

3. A FRAME-EDGE EXCLUSION LINE.
   The user's hypothesis, and the evidence fits: v1 said "report every player
   you can see, including partly hidden ones" and gave no instruction on how to
   box one, so a model shown half a head was told to report a player and left
   to imagine where the rest of the body goes. That is exactly a "fly-in" - the
   marker wrong on a player's entry frame and correct once the body is visible.
   v2 added "box only the part you can actually see"; v3 goes further and
   declines the player altogether while they are mostly outside the frame.
   This HAS A REAL COST - genuine players at the margins go unreported and
   marker counts fall - which is why it is a separate arm and not a silent
   change.

NOT DONE, deliberately: sparse `num`. Moving jersey numbers out of the row into
a frame-level map saves ~6% of the array, but couples every number to an array
INDEX - so one dropped or malformed row silently reassigns numbers to the wrong
players. That is a new and much worse failure mode than the one it saves, for
about 2% of output tokens. Raised rather than implemented.
"""

FRACTION, THOUSANDTH = "fraction", "thousandth"

COORD_SPACE_V3 = THOUSANDTH

PROMPT_V3 = """One frame of sports footage. Report every player on the playing
surface, and the ball.

Do NOT report: referees and other match officials, substitutes and anyone on the
bench, coaches, medical staff, the crowd, ball boys.

Do NOT report a player who is more than half outside the frame. If you can see
most of them, report them and box only the part you can actually see. Report
exactly as many players as you can see - there is no expected number and it does
not depend on the sport."""

_COORD_NOTE = ("as a whole number from 0 to 1000, where 0 is the left edge of "
               "the image and 1000 the right")
_COORD_NOTE_Y = ("as a whole number from 0 to 1000, where 0 is the top edge of "
                 "the image and 1000 the bottom")

SCHEMA_V3 = {
    "name": "frame_detections",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["scene", "players", "ball"],
        "properties": {
            "scene": {
                "type": "string",
                "description": "One short sentence: camera framing, lighting, "
                               "and the shirt colour of each team. Written "
                               "BEFORE any coordinates, as working-out."
            },
            "players": {
                "type": "array",
                "description": "One entry per player on the field of play, "
                               "GOALKEEPERS INCLUDED. Box tightly: top edge at "
                               "the crown of the head, bottom edge where the "
                               "feet meet the ground. Empty array is valid and "
                               "correct if none are visible.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x", "y", "w", "h", "kit", "num"],
                    "properties": {
                        "x": {"type": "integer",
                              "description": f"LEFT edge of the box, {_COORD_NOTE}"},
                        "y": {"type": "integer",
                              "description": f"TOP edge of the box, {_COORD_NOTE_Y}"},
                        "w": {"type": "integer",
                              "description": "box width, in the same 0-1000 "
                                             "units as x"},
                        "h": {"type": "integer",
                              "description": "box height, in the same 0-1000 "
                                             "units as y"},
                        "kit": {"type": "string",
                                "description": "Shirt colour as one common "
                                               "word. Judge the colour itself; "
                                               "never \"team A\" or \"home\" - "
                                               "each frame is judged on its own "
                                               "and the words must agree across "
                                               "frames."},
                        "num": {"type": ["integer", "null"],
                                "description": "Jersey number ONLY if you can "
                                               "actually read it. null "
                                               "otherwise. A wrong number is "
                                               "far worse than no number."},
                    }
                }
            },
            "ball": {
                "type": ["object", "null"],
                "description": "null when the ball is not visible, which it "
                               "often is not. The ball is ABOVE the playing "
                               "surface: not a mark painted on it, and not "
                               "something a player is wearing or carrying. "
                               "Never place it where you think it ought to be.",
                "additionalProperties": False,
                "required": ["x", "y", "w", "h", "conf"],
                "properties": {
                    "x": {"type": "integer", "description": f"left edge, {_COORD_NOTE}"},
                    "y": {"type": "integer", "description": f"top edge, {_COORD_NOTE_Y}"},
                    "w": {"type": "integer", "description": "width, 0-1000 units"},
                    "h": {"type": "integer", "description": "height, 0-1000 units"},
                    "conf": {"type": "number",
                             "description": "0.0 to 1.0, how sure you are"}
                }
            }
        }
    }
}

COMPACT_ORDER_V3 = ["x", "y", "w", "h", "kit", "num"]
COMPACT_SCHEMA_V3 = {
    "name": "frame_detections_compact",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["scene", "players", "ball"],
        "properties": {
            "scene": {"type": "string",
                      "description": "One short sentence: camera framing, "
                                     "lighting, and the shirt colour of each "
                                     "team. Written BEFORE any coordinates, as "
                                     "working-out."},
            "players": {
                "type": "array",
                # The array form loses per-field typing, so this sentence is the
                # only thing carrying the contract - it states the type and the
                # UNITS of every position, and the units are the whole change.
                "description": ("One array per player, GOALKEEPERS INCLUDED, "
                                "ALWAYS in this order: [x, y, w, h, kit, num]. "
                                "All four numbers are WHOLE NUMBERS from 0 to "
                                "1000, never decimals: 0 is the left/top edge "
                                "of the image and 1000 the right/bottom. "
                                "x = LEFT edge of the box (integer). "
                                "y = TOP edge of the box (integer). "
                                "w = box width in the same units (integer). "
                                "h = box height in the same units (integer). "
                                "Box tightly: top edge at the crown of the "
                                "head, bottom edge where the feet meet the "
                                "ground. "
                                "kit = shirt colour as one ordinary word "
                                "(string): red, blue, white, yellow, green, "
                                "black, orange, purple. Judge the colour "
                                "itself; never \"team A\" or \"home\", because "
                                "each frame is judged on its own and the words "
                                "must agree across frames. "
                                "num = the number on the shirt as an integer, "
                                "ONLY if you can genuinely read it, otherwise "
                                "null. A wrong number is far worse than no "
                                "number. "
                                "An empty array is valid and correct if no "
                                "players are visible."),
                "items": {"type": "array",
                          "items": {"type": ["integer", "string", "null"]}}},
            "ball": {"type": ["array", "null"],
                     "description": ("[x, y, w, h, conf] or null, boxed "
                                     "tightly. x, y, w and h are WHOLE NUMBERS "
                                     "from 0 to 1000 on the same scale as the "
                                     "players; conf is a decimal 0.0 to 1.0. "
                                     "The ball is ABOVE the playing surface: "
                                     "not a mark painted on it, and not "
                                     "something a player is wearing or "
                                     "carrying. If you cannot see the ball, "
                                     "null - null is common and correct. Never "
                                     "place it where you think it ought to be."),
                     "items": {"type": ["number", "null"]}},
        }}}
