def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2) - 1
        return listnum[i]
    else:
        i = int(lnum / 2) - 1
        return (listnum[i] + listnum[i + 1]) / 2


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


throughput_pu = []
throughput_greedy = []
throughput_heuristic = []
F_pu = []
F_greedy = []
F_heuristic = []
average_pu = []
average_greedy = []
average_heuristic = []
exp_num = 200
# C:/Users/Administrator/Desktop/Experimental data set/nodes_number_ratio/result_120_Greedy.txt'
with open('response_v1/node_failure_probability/result_0.30_Pu.txt', mode='r') as f_pu:
    for i in range(exp_num):
        # line1 = f_pu.readline().replace('Average Capacity Utilization:', '').replace('\n', '')
        # f_pu.readline()
        # throughput_pu.append(float(line1))

        # line1 = f_pu.readline().replace('Weighted Throughput:', '').replace('\n', '')
        # line2 = f_pu.readline().replace('Throught which >= F_th: ', '').replace('\n', '')
        # f_pu.readline()
        # throughput_pu.append(float(line1))
        # F_pu.append(float(line2))

        line1 = f_pu.readline().replace('Weighted Throughput:', '').replace('\n', '')
        line2 = f_pu.readline().replace('Throught which >= F_th: ', '').replace('\n', '')
        f_pu.readline()
        f_pu.readline()
        line3 = f_pu.readline().replace('Average Capacity Utilization:', '').replace('\n', '')
        f_pu.readline()
        throughput_pu.append(float(line1))
        F_pu.append(float(line2))
        average_pu.append(float(line3))

        with open('E:/xzr/论文/Communication Physics-网络级的纯化决策设计/实验代码以及相关实验数据/Experimental data '
                  'set/node_failure_probability_throughput/result_0.30_PU.txt',
                  mode='a') as ffff:
            ffff.write('Weighted Throughput:' + str(float(line1)))
            ffff.write('\n')
            ffff.write('Throught which >= F_th:' + str(float(line2)))
            ffff.write('\n')
            ffff.write('##########################################################################################')
            ffff.write('\n')

            # ffff.write('Average Capacity Utilization:' + str(round(float(line3), 2)))
            # ffff.write('\n')
            # ffff.write('##########################################################################################')
            # ffff.write('\n')


    # throughput_pu = throughput_pu + float(line1)
        # F_pu = F_pu + float(line2)
        # average_pu = average_pu + float(line3)

with open('response_v1/node_failure_probability/result_0.30_Greedy.txt',
          mode='r') as f_greedy:
    for i in range(exp_num):
        line1 = f_greedy.readline().replace('Weighted Throughput:', '').replace('\n', '')
        line2 = f_greedy.readline().replace('Throught which >= F_th: ', '').replace('\n', '')
        f_greedy.readline()
        f_greedy.readline()
        line3 = f_greedy.readline().replace('Average Capacity Utilization:', '').replace('\n', '')
        f_greedy.readline()
        throughput_greedy.append(float(line1))
        F_greedy.append(float(line2))
        average_greedy.append(float(line3))

        with open('E:/xzr/论文/Communication Physics-网络级的纯化决策设计/实验代码以及相关实验数据/Experimental data '
                  'set/node_failure_probability_throughput/result_0.30_Greedy.txt',
                  mode='a') as ffff:
            ffff.write('Weighted Throughput:' + str(float(line1)))
            ffff.write('\n')
            ffff.write('Throught which >= F_th:' + str(float(line2)))
            ffff.write('\n')
            ffff.write('##########################################################################################')
            ffff.write('\n')

            # ffff.write('Average Capacity Utilization:' + str(round(float(line3), 2)))
            # ffff.write('\n')
            # ffff.write('##########################################################################################')
            # ffff.write('\n')

with open('response_v1/node_failure_probability/result_0.30_Heuristic.txt',
          mode='r') as f_heuristic:
    for i in range(exp_num):
        # f_heuristic.readline()
        line1 = f_heuristic.readline().replace('Weighted Throughput:', '').replace('\n', '')
        line2 = f_heuristic.readline().replace('Throught which >= F_th: ', '').replace('\n', '')
        f_heuristic.readline()
        f_heuristic.readline()
        line3 = f_heuristic.readline().replace('Average Capacity Utilization:', '').replace('\n', '')
        f_heuristic.readline()
        throughput_heuristic.append(float(line1))
        F_heuristic.append(float(line2))
        average_heuristic.append(float(line3))

        with open('E:/xzr/论文/Communication Physics-网络级的纯化决策设计/实验代码以及相关实验数据/Experimental data '
                  'set/node_failure_probability_throughput/result_0.30_Heuristic.txt',
                  mode='a') as ffff:
            ffff.write('Weighted Throughput:' + str(float(line1)))
            ffff.write('\n')
            ffff.write('Throught which >= F_th:' + str(float(line2)))
            ffff.write('\n')
            ffff.write('##########################################################################################')
            ffff.write('\n')

            # ffff.write('Average Capacity Utilization:' + str(round(float(line3), 2)))
            # ffff.write('\n')
            # ffff.write('##########################################################################################')
            # ffff.write('\n')

# print('pu##########################################pu')
# print(mediannum(throughput_pu))
# print(mediannum(F_pu))
# print(mediannum(average_pu))
# print('greedy##########################################greedy')
# print(mediannum(throughput_greedy))
# print(mediannum(F_greedy))
# print(mediannum(average_greedy))
# print('heuristic##########################################heuristic')
# print(mediannum(throughput_heuristic))
# print(mediannum(F_heuristic))
# print(mediannum(average_heuristic))
print('pu##########################################pu')
print(averagenum(throughput_pu))
print(averagenum(F_pu))
print(averagenum(average_pu))
print('greedy##########################################greedy')
print(averagenum(throughput_greedy))
print(averagenum(F_greedy))
print(averagenum(average_greedy))
print('heuristic##########################################heuristic')
print(averagenum(throughput_heuristic))
print(averagenum(F_heuristic))
print(averagenum(average_heuristic))
