class CastingDataReader():
    def __init__(self) -> None:
        pass

    def Read():
        file_path = "../datasets/preprocessed/casting/2021-04-07-17-54-00-strand-1.csv"
        file = open(file_path)
        lines = file.readlines()

        # 液位能被检测之后的数据
        hs = list()
        ts = list()
        ls = list()

        # 液位能被检测之前的数据
        phs = list()
        pts = list()

        # completed data
        chs = list()
        cts = list()

        is_header_passed = False
        is_lv_detected = False  # 在结晶器中的钢液是否能被检测到
        ready_to_start = False

        sensor_to_dummy_bar_height = 350

        for line in lines:
            if is_header_passed:
                nums = line.split(',')
                current_l = float(nums[1])
                if is_lv_detected:
                    hs.append(float(nums[0]))
                    ls.append((float(nums[1]) + sensor_to_dummy_bar_height))
                    ts.append(0.5)
                    chs.append(float(nums[0]))
                    cts.append(0.5)
                if ready_to_start and not is_lv_detected:
                    pre_lv_act = float(nums[1]) + sensor_to_dummy_bar_height
                    is_lv_detected = True
                if current_l > 2:
                    ready_to_start = True
                else:
                    chs.append(float(nums[0]))
                    cts.append(0.5)
                    phs.append(float(nums[0]))
                    pts.append(0.5)
            else:
                is_header_passed = True
        return (hs, ls, ts, phs, pts, pre_lv_act, chs, cts)