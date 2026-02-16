"""
Extracts images in bytes from an MCAP file for the 4 camera topics (center, center, left, right).
Similar to orientation.py: registers schemas, iterates messages, saves a CSV index and writes the bytes to disk.
"""
import re
from pathlib import Path

import pandas as pd
from mcap.reader import make_reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import Stores, get_typestore, get_types_from_msg, get_types_from_idl

path = "2025-10-10_19-22_normal_0.mcap"
# 4 topics just like in dataset_builder (center twice, left, right)
CAMERA_TOPICS = [
    "/lucid_vision/lucid_cam_front_center/image_rect/compressed",
    "/lucid_vision/lucid_cam_front_center/image_rect/compressed",
    "/lucid_vision/lucid_cam_front_left/image_rect/compressed",
    "/lucid_vision/lucid_cam_front_right/image_rect/compressed",
]
TOPICS_SET = set(CAMERA_TOPICS)
OUT_DIR = Path("images_extracted")  # Each topic will have a subfolder where images are saved as <t_ns>.jpg

typestore = get_typestore(Stores.ROS2_FOXY)

def topic_to_slug(topic: str) -> str:
    # Convert ROS topic name to a filesystem-safe slug
    return re.sub(r"[^a-zA-Z0-9]", "_", topic.replace("/", "_").strip("_"))

def stamp_to_float(t) -> float:
    # Convert ROS stamp message to float seconds
    return float(t.sec) + float(t.nanosec) * 1e-9

rows = []

with open(path, "rb") as f:
    reader = make_reader(f)
    summary = reader.get_summary()

    # Register ROS message schemas found in the MCAP summary
    for _, schema in summary.schemas.items():
        enc = getattr(schema, "encoding", None) or getattr(schema, "schema_encoding", None)
        if enc not in ("ros2msg", "ros2idl"):
            continue
        text = schema.data.decode("utf-8", errors="replace")
        try:
            if enc == "ros2msg":
                types = get_types_from_msg(text, schema.name)
            else:
                types = get_types_from_idl(text)
            typestore.register(types)
        except Exception:
            # Ignore schemas that fail to register
            pass

    # Iterate through all messages in the MCAP file
    for schema, channel, msg in reader.iter_messages():
        # Only process messages from the desired camera topics with correct encoding and type
        if channel.topic not in TOPICS_SET:
            continue
        if channel.message_encoding != "cdr":
            continue
        if schema.name != "sensor_msgs/msg/CompressedImage":
            continue

        try:
            # Deserialize the message to get image data
            data = deserialize_cdr(msg.data, schema.name, typestore)
        except Exception:
            # Skip if deserialization fails
            continue

        t = data.header.stamp
        ts = stamp_to_float(t)
        t_ns = msg.log_time
        # Save the timestamp of the first message for elapsed time calculations
        if not rows:
            t0 = ts
        elapsed = ts - rows[0]["t"] if rows else 0.0
        img_bytes = bytes(data.data)
        fmt = getattr(data, "format", "jpeg") or "jpeg"
        ext = "jpg" if "jpeg" in fmt.lower() else "png"

        # Create output subdirectory for this topic and write the image file
        slug = topic_to_slug(channel.topic)
        out_sub = OUT_DIR / slug
        out_sub.mkdir(parents=True, exist_ok=True)
        out_path = out_sub / f"{t_ns}.{ext}"
        out_path.write_bytes(img_bytes)

        # Record metadata for this image
        rows.append({
            "t": ts,
            "t_ns": t_ns,
            "sec": int(t.sec),
            "elapsed": elapsed,
            "nanosec": int(t.nanosec),
            "topic": channel.topic,
            "format": fmt,
            "path": str(out_path),
            "size_bytes": len(img_bytes),
        })

# Build DataFrame and save the table as a CSV index
df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
df.to_csv(OUT_DIR / "images_index.csv", index=False)
print("Images:", len(df), "->", OUT_DIR)
