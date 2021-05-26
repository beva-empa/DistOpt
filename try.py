exp_list = []
imp_list = []
hub_list = []
for id in range(num_hubs):
    exp_list.append([])
    imp_list.append([])

j = 0
for id in range(num_hubs):
    hub_list.append((id+1)*(num_hubs-1))
    for kd in range(num_hubs):
        if kd != id:
            exp_list[id].extend([j])
            j += 1

start = -1
for i in range(num_hubs):
    j = 0
    for k in range(num_hubs):
        if k == i:
            j = 1
        else:
            imp_list[i].extend([start+j+(num_hubs-1)*k])
    start += 1
