from datetime import datetime, timezone,timedelta

def convert_to_integer(date_string):
    date_object = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    timestamp = int(date_object.replace(tzinfo=timezone.utc).timestamp())*(10 ** 3)
    return timestamp

def generate_5_minute_intervals(start_time):
    intervals = []
    base_time = datetime.strptime(start_time, "%Y%m%d_%H%M")
    for i in range(12):  # 12 intervals of 5 minutes in an hour
        interval_time = base_time + timedelta(minutes=5 * i)
        intervals.append(interval_time.strftime("%Y-%m-%d %H:%M:%S"))
    return intervals

def generate_intervals(start_time, num_intervals):
    interval_list = []
    current_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    for i in range(num_intervals):
        interval_list.append(current_time.strftime('%Y-%m-%d %H:%M:%S'))
        current_time += timedelta(minutes=5)
    return interval_list

def map_hour_to_windows(hour, interval_length, overlap):
    windows = []
    num_windows = 24 // (interval_length - overlap)  # Calculate number of windows needed
    for i in range(num_windows):
        start = i * (interval_length - overlap)
        end = start + interval_length
        if start <= hour < end or (start <= hour + 24 < end):  # Handle wrap-around for hours
            windows.append(i + 1)
    return windows
