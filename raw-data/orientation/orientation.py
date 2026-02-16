import pandas as pd
from mcap.reader import make_reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_typestore, Stores
from rosbags.typesys import get_types_from_msg, get_types_from_idl

path = "2025-10-10_19-22_normal_0.mcap"
topic_target = "/imu/data"

typestore = get_typestore(Stores.ROS2_FOXY)

def stamp_to_float(t):
    # Convert stamp to float seconds
    return float(t.sec) + float(t.nanosec) * 1e-9

rows = []

with open(path, "rb") as f:
    reader = make_reader(f)
    summary = reader.get_summary()

    # Register custom embedded ROS schemas found in the MCAP summary
    for _, schema in summary.schemas.items():
        schema_encoding = getattr(schema, "encoding", None) or getattr(schema, "schema_encoding", None)
        if schema_encoding not in ("ros2msg", "ros2idl"):
            continue

        text = schema.data.decode("utf-8", errors="replace")
        try:
            if schema_encoding == "ros2msg":
                types = get_types_from_msg(text, schema.name)
            else:
                types = get_types_from_idl(text)
            typestore.register(types)
        except Exception:
            # Ignore errors during schema registration
            pass

    # Iterate through all messages in the MCAP file
    for schema, channel, msg in reader.iter_messages():
        # Filter messages for the target IMU topic and check encoding
        if channel.topic != topic_target:
            continue
        if channel.message_encoding != "cdr":
            continue

        # Deserialize message to access orientation data
        data = deserialize_cdr(msg.data, schema.name, typestore)

        # Parse the timestamp
        t = data.header.stamp
        ts = stamp_to_float(t)

        # Calculate elapsed time relative to first message
        if not rows:
            t0 = ts
        elapsed = ts - rows[0]["t"] if rows else 0.0

        # Store the relevant orientation and timing data
        rows.append({
            "t": ts,
            "sec": int(t.sec),
            "elapsed": elapsed,
            "nanosec": int(t.nanosec),
            "orientation.x": float(data.orientation.x),
            "orientation.y": float(data.orientation.y),
            "orientation.z": float(data.orientation.z),
            "orientation.w": float(data.orientation.w),
        })

# Build DataFrame, sort, and reset index
df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)

# Write orientation data to CSV file
df.to_csv("orientation.csv", index=False)  