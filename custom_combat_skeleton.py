"""MMPose dataset metadata for the COCO-17 combat-sports skeleton."""

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SWAPS = {
    "left_eye": "right_eye",
    "right_eye": "left_eye",
    "left_ear": "right_ear",
    "right_ear": "left_ear",
    "left_shoulder": "right_shoulder",
    "right_shoulder": "left_shoulder",
    "left_elbow": "right_elbow",
    "right_elbow": "left_elbow",
    "left_wrist": "right_wrist",
    "right_wrist": "left_wrist",
    "left_hip": "right_hip",
    "right_hip": "left_hip",
    "left_knee": "right_knee",
    "right_knee": "left_knee",
    "left_ankle": "right_ankle",
    "right_ankle": "left_ankle",
}

SKELETON_LINKS = [
    ("left_ankle", "left_knee"),
    ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"),
    ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_eye", "right_eye"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
]


def _side(name: str) -> str:
    return "lower" if any(part in name for part in ("hip", "knee", "ankle")) else "upper"


def _color(name: str) -> list[int]:
    if name.startswith("left_"):
        return [0, 255, 0]
    if name.startswith("right_"):
        return [255, 128, 0]
    return [51, 153, 255]


dataset_info = dict(
    dataset_name="combat_coco17",
    paper_info=dict(
        author="Combat sports dataset contributors",
        title="UFC/MMA Fighter Keypoints",
        container="Custom dataset",
        year="2026",
        homepage="",
    ),
    keypoint_info={
        index: dict(
            name=name,
            id=index,
            color=_color(name),
            type=_side(name),
            swap=SWAPS.get(name, ""),
        )
        for index, name in enumerate(KEYPOINT_NAMES)
    },
    skeleton_info={
        index: dict(link=link, id=index, color=[51, 153, 255])
        for index, link in enumerate(SKELETON_LINKS)
    },
    joint_weights=[
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.2,
        1.2,
        1.5,
        1.5,
        1.0,
        1.0,
        1.2,
        1.2,
        1.5,
        1.5,
    ],
    sigmas=[
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ],
)
