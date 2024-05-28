from datetime import datetime


def convert_to_timestamp(start, end):
    # time_select = '2024-05-06 08:00'

    start_dt_object = datetime.strptime(start, '%Y-%m-%d %H:%M')
    st_timestamp = int(start_dt_object.timestamp() * 1000)

    end_dt_object = datetime.strptime(end, '%Y-%m-%d %H:%M')
    end_timestamp = int(end_dt_object.timestamp() * 1000)
    return st_timestamp, end_timestamp


start = '2024-05-04 08:00'
end = '2024-05-04 10:00'
st_timestamp, end_timestamp = convert_to_timestamp(start, end)

print("start_time:", st_timestamp)
print("end_time:", end_timestamp)

