from data_segments.get_data_segments import DataSegment, get_segments_v2

segments = get_segments_v2()
for session, data_type, start_ms, stop_ms in segments:
    if data_type == "test":
        segment = DataSegment(session, data_type, start_ms, stop_ms)
        segment.get_flame_params("P1", "voca", only_odd=True)
        segment.get_flame_params("P2", "voca", only_odd=True)

        
segment = DataSegment("35", "test", 0, 1001340)
segment.get_flame_params("P1", "voca", only_odd=True)
segment.get_flame_params("P2", "voca", only_odd=True)