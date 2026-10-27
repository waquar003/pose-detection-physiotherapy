from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

MP_MODEL_PATH = MODELS_DIR / "pose_landmarker.task"

MLP_WEIGHTS = {
    "C17": MODELS_DIR / "mapper_model_C17.pth",
    "C18": MODELS_DIR / "mapper_model_C18.pth",
    "combined": MODELS_DIR / "mapper_model_combined.pth",
}
DEFAULT_MODEL = "combined"

EXERCISES = {
    "Ex1": "Ex1 - Arm Abduction",
    "Ex2": "Ex2 - Arms V/W Shape",
    "Ex3": "Ex3 - Push-up",
    "Ex4": "Ex4 - Standing Side Leg Raise",
    "Ex5": "Ex5 - Lunge",
    "Ex6": "Ex6 - Squat",
}

GHOST_MODELS = {
    "Ex1": MODELS_DIR / "Ex1_A9_ens_hmm_sdtw_winreg.pkl",
    "Ex2": MODELS_DIR / "Ex2_A9_ens_hmm_sdtw_winreg.pkl",
    "Ex3": MODELS_DIR / "Ex3_A4_winreg_clamp_cnn.pkl",
    "Ex4": MODELS_DIR / "Ex4_A4_winreg_clamp_cnn.pkl",
    "Ex5": MODELS_DIR / "Ex5_A9_ens_hmm_sdtw_winreg.pkl",
    "Ex6": MODELS_DIR / "Ex6_A10_embed_clamp_embnn.pkl",
}

MP_MIN_DETECTION_CONFIDENCE = 0.4
VIS_THR = 0.1                      # landmark group counts as hidden below this mean visibility

# The models were trained on UNMIRRORED frames: detect on the raw frame, mirror only the display.
MIRROR_INPUT = False
MIRROR_DISPLAY = True

COACH_FPS = 30.0                   # models were trained on 30 fps reps; live joints are resampled to this rate

JOINT26_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (6, 7), (7, 8), (8, 9), (11, 12), (12, 13), (13, 14),
    (0, 6), (0, 11), (1, 6), (1, 11),
    (0, 16), (0, 21), (16, 17), (17, 18), (18, 19), (19, 20), (21, 22), (22, 23), (23, 24), (24, 25),
    (4, 5), (9, 10), (14, 15),
]


# Skeleton / Features
JN = {"Hips": 0, "Spine": 1, "Spine1": 2, "Neck": 3, "Head": 4, "Head_end": 5,
      "LeftShoulder": 6, "LeftArm": 7, "LeftForeArm": 8, "LeftHand": 9, "LeftHand_end": 10,
      "RightShoulder": 11, "RightArm": 12, "RightForeArm": 13, "RightHand": 14, "RightHand_end": 15,
      "LeftUpLeg": 16, "LeftLeg": 17, "LeftFoot": 18, "LeftToeBase": 19, "LeftToeBase_end": 20,
      "RightUpLeg": 21, "RightLeg": 22, "RightFoot": 23, "RightToeBase": 24, "RightToeBase_end": 25}
JN_NAMES = [n for n, _ in sorted(JN.items(), key=lambda kv: kv[1])]

ARM_J = ["Neck", "Head", "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand", "Spine1"]
LEG_J = ["Neck", "Head", "Spine1", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
         "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]
FULL_J = sorted(set(ARM_J + LEG_J))
EX_JOINTS = {"Ex1": ARM_J, "Ex2": ARM_J, "Ex3": FULL_J, "Ex4": LEG_J, "Ex5": LEG_J, "Ex6": LEG_J}

ANGLE_DEFS = [("Neck", "LeftShoulder", "LeftArm"), ("Neck", "RightShoulder", "RightArm"),
              ("LeftShoulder", "LeftArm", "LeftForeArm"), ("RightShoulder", "RightArm", "RightForeArm"),
              ("LeftArm", "LeftForeArm", "LeftHand"), ("RightArm", "RightForeArm", "RightHand"),
              ("Hips", "Spine1", "Neck"),
              ("LeftUpLeg", "LeftLeg", "LeftFoot"), ("RightUpLeg", "RightLeg", "RightFoot"),
              ("Spine", "LeftUpLeg", "LeftLeg"), ("Spine", "RightUpLeg", "RightLeg"),
              ("LeftLeg", "LeftFoot", "LeftToeBase"), ("RightLeg", "RightFoot", "RightToeBase")]

ANG_NAMES = ["neck_shL", "neck_shR", "armL", "armR", "elbowL", "elbowR", "trunk", "kneeL", "kneeR", "hipL", "hipR",
             "ankleL", "ankleR", "shoulder_tilt", "hip_tilt", "trunk_lean", "head_lean"]

def _ai(*n): 
    return [ANG_NAMES.index(x) for x in n]

_ARM = _ai("neck_shL", "neck_shR", "armL", "armR", "elbowL", "elbowR", "trunk", "shoulder_tilt", "trunk_lean")
_LEG = _ai("trunk", "kneeL", "kneeR", "hipL", "hipR", "hip_tilt", "trunk_lean")

EX_ANG = {"Ex1": _ARM, "Ex2": _ARM,
          "Ex3": _ai("armL", "armR", "elbowL", "elbowR", "trunk", "kneeL", "kneeR", "hipL", "hipR", "trunk_lean", "head_lean"),
          "Ex4": _LEG, "Ex5": _LEG, "Ex6": _LEG}

_ANGLE_JOINTS = [set(t) for t in ANGLE_DEFS] + [{"LeftShoulder", "RightShoulder"}, {"LeftUpLeg", "RightUpLeg"},
                                                 {"Neck", "Hips"}, {"Head", "Neck"}]

EMA_ALPHA, VEL_LAG = 0.5, 5
WIN_BACK, WIN_FWD = 0.08, 0.12
K0, KV, PCONF_MIN, GATE_NU, COAST_RELOCK = 0.35, 0.05, 0.35, 0.30, 15