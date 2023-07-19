def count_number_in_intervals(intervals, numbers):
    # intervals = [1, 0.1, 0.01, 0.001, 0.0001]
    interval_counts = [0] * len(intervals)

    for num in numbers:
        for i, interval in enumerate(intervals):
            if 0 <= num < interval:
                continue
            interval_counts[i] += 1
            break

    return interval_counts


if __name__ == '__main__':
    # numbers = [0.05, 0.0012, 0.023, 0.0005, 0.00008, 0.1, 0.002, 0.0009]
    intervals = [0.1, 0.01, 0.008, 0.006, 0.002, 0.001, 0.0001]
    json_path = 'D:\seekoo\SD\sd-scripts\cones\e30-K30-th0-colorful_rhythm-concept_neurons.json'
    import json
    from tqdm import tqdm
    with open(json_path, 'r')as json_file:
        data = json.load(json_file)

    Mp_list = []
    for lora_name, param_dict in tqdm(data.items()):
        for param_name, cone in param_dict.items():
            Mp_list.append(cone['Mp'])

    result = count_number_in_intervals(intervals, Mp_list)
    for i, interval in enumerate(intervals):
        if i != 0:
            print(f"There are {result[i]} Mp ~ {interval}->{intervals[i-1]}")
        else:
            print(f"There are {result[0]} Mp ~ > {interval}")
