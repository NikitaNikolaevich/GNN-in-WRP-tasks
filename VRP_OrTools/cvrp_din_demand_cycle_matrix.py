# Подбор макропараметров перебором

#import my_plots
from cvrp_ortools_base import solve_cvrp
from matplotlib import pyplot as plt
import pickle
import random
from datetime import datetime
import pandas as pd

matrix_data_path = "./datasets/matrices_by_hour_40_nodes"

# Получить список узлов потребления
with open(f'{matrix_data_path}/matrix_mon_0000_40_nodes.pkl', 'rb') as f:
    basic_matrix_df = pickle.load(f)
chosen_points = [str(i) for i in basic_matrix_df.columns][1:]

# Получить распределенные данные спроса за выбранную неделю
demands = pd.read_csv("datasets/second_scenario.csv")
demands.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
demands['date'] = pd.to_datetime(demands['date'])
# Устанавливаем 'date' в качестве индекса
demands.set_index('date', inplace=True)
demands = demands.loc['2023-07-01 00:00:00':'2023-07-07 23:00:00']

# Стоимостные характеристики
# Стоимость аренды ТС в сутки
rent_cost = {'4000': 1500, '5000': 1800, '6000': 2000, '8000': 2500, '10000': 3000}
# Стоимость минуты движения ТС
distance_cost = {'4000': 3, '5000': 3.5, '6000': 4, '8000': 5, '10000': 6}

# Функция для добавления шума в спрос по часам
def add_noise(demand: list, noise_type: str = 'небольшое повышение спроса'):
    # noise_type - 'небольшое повышение спроса', 'всплеск спроса'
    if noise_type=='небольшое повышение спроса':
        demand = [round(i*(1+random.randint(0, 0)/100)) for i in demand]
    elif noise_type=='всплеск спроса':
        demand = [round(i*(1+random.randint(0, 30)/100)) for i in demand]
    return demand

# Проверка, не упал ли запас в точках до нуля
def check_zero_stock(din_stock, hour):
    current_stock = [din_stock[k][hour] for k in din_stock.keys()]
    if any(d <= 0 for d in current_stock):
        print('\n ОБНУЛИЛСЯ ЗАПАС!')
        return True
    else:
        return False

# Получить потенциальное распределение запасов с текущего часа (если больше не будет доставки)
def get_potential_stock(din_stock, timer):
    from_now_stock = {}
    for node in din_stock.keys():
        from_now_stock[node] = din_stock[node][timer:]
    return from_now_stock

# Известны ли при планировании фактические запасы в узлах
check_fact_stock = False

# Вывод логов на печать
to_print = True
# Set node to start
start_node = '0'

# Задать прогноз спроса
demand_predictions = {}
for i in range(len(chosen_points)):
    demand_predictions[chosen_points[i]] = demands[str(i)].to_list()

# Задать факт спроса
demand_fact = {}
for node_key in demand_predictions:
    demand_fact[node_key] = add_noise(demand_predictions[node_key], 'небольшое повышение спроса')

def vrp_for_period(vehicles: int, capacity: int, warehouse_stock: int, init_stock: int, timer_pace: int,
                   predict_ahead: int, soft_limit: float, first: str, search_option, search_time, to_print=True):

    # Для визуализации маршрутов
    vis_results = []
    hard_limit = 0.1
    start_time = datetime.now()

    total_routing_time = 0
    routes = 0 # Сколько раз решалась задача VRP/планировался выезд ТС за период

    max_time = len(demand_predictions[chosen_points[0]]) - predict_ahead

    # Установить начальное состояние машин
    all_vehicles = {}
    for v in range(1, vehicles + 1):
        all_vehicles[str(v)] = {'in_depo': True, 'returns_in': 0}

    # Установить размер склада в узлах и начальные запасы
    warehouses = {}
    init_stocks = {}
    for point in chosen_points:
        warehouses[point] = warehouse_stock
        init_stocks[point] = init_stock

    # Рассчитать размер уменьшения запасов по мере потребления по часам (если бы ничего не довозили) с ориентацией на прогноз спроса
    din_stock_predictions = {}
    for k in demand_predictions.keys():
        din_stock_predictions[k] = [init_stocks[k] - sum(demand_predictions[k][:i + 1]) for i in
                                    range(len(demand_predictions[k]))]

    # Рассчитать размер уменьшения запасов по мере потребления по часам (если бы ничего не довозили) с ориентацией на фактический спрос
    din_stock_fact = {}
    for k in demand_predictions.keys():
        din_stock_fact[k] = [init_stocks[k] - sum(demand_fact[k][:i + 1]) for i in range(len(demand_fact[k]))]

    # Статистика движения ТС для расчета стоимости
    vehicles_stats = {}

    print(f'Прогноз на период в часах: {predict_ahead}, Периодичность планирования в часах: {timer_pace}')

    for timer in range(max_time // timer_pace):
        timer = timer * timer_pace
        the_day = timer//24 # сутки планирования
        the_hour = timer - 24*(the_day) # час суток
        if f'день_{str(the_day)}' not in vehicles_stats.keys():
            vehicles_stats[f'день_{str(the_day)}'] = {}

        if to_print:
            print(f'\nЧас {timer} ------------------------------------------------------------------------------------')
        # Проверить, не обнулился ли где-то запас ПО ФАКТУ (в этом случае задача считается нерешенной)
        fatal_stock = check_zero_stock(din_stock_fact, timer)

        if fatal_stock:
            return 0,0,0,0

        if to_print:
            print('Состояние машин:')
            print(all_vehicles)

        # print('\nСостояние запасов ПО ФАКТУ до конца дня, если доставки после этого часа больше не будет:')
        # from_now_stock_fact = get_potential_stock(din_stock_fact, timer)
        # for k in from_now_stock_fact.keys():
        #     print(k, ' ', from_now_stock_fact[k])
            
        #print('\nСостояние запасов ПО ПРОГНОЗУ до конца дня, если доставки после этого часа больше не будет:')
        #from_now_stock_pred = get_potential_stock(din_stock_predictions, timer)
        #for k in from_now_stock_pred.keys():
        #    print(k, ' ', from_now_stock_pred[k])
            

        if check_fact_stock:
            # Обновить прогноз падения запаса с учетом знаний о том, что текущий запас может отличаться от прогнозного на текущий час
            for node in list(din_stock_predictions.keys()):
                delta_stock = din_stock_predictions[node][timer] - din_stock_fact[node][timer]
                if delta_stock != 0:
                    din_stock_predictions[node] = [(din_stock_predictions[node][i] - delta_stock) if i >= timer
                                                   else din_stock_predictions[node][i] for i in
                                                   range(len(din_stock_predictions[node]))]

        # Вычислить, не упадёт ли запас ПО ПРОГНОЗУ в некоторых точках до "жесткого" порогового значения (10% склада) в течение n часов
        # (критические точки)
        future_demand = {}
        for node in din_stock_predictions.keys():
            future_demand[node] = max(warehouses[node]*hard_limit - din_stock_predictions[node][timer + predict_ahead], 0)
        # Если есть спрос, запустить задачу VRP и отправить машины на доставку
        if any(v > 0 for v in list(future_demand.values())):
            vis_vrp = []
            if to_print:
                print(f'\nВ течение {predict_ahead} часов ПО ПРОГНОЗУ запас упадёт до критического значения в узлах:')
                print([n for n in future_demand.keys() if future_demand[n] > 0])
            # Определить точки со спросом (не только критические) ПО ПРОГНОЗУ, и количество груза для доставки
            current_demand = {}
            for node in din_stock_predictions.keys():
                current_demand[node] = max(warehouses[node]*soft_limit - din_stock_predictions[node][timer + predict_ahead], 0)

            # Определить точки со спросом (не только критические) ПО ФАКТУ, и количество груза для доставки
            current_demand_fact = {}
            for node in din_stock_fact.keys():
                current_demand_fact[node] = max(warehouses[node]*soft_limit - din_stock_fact[node][timer + predict_ahead], 0)

            # Получить матрицу расстояний только для узлов, которые будут участвовать в доставке (узлы со спросом и депо)
            nodes_with_demand = [n for n in current_demand.keys() if current_demand[n] > 0]
            if to_print:
                print('\nПО ПРОГНОЗУ есть спрос в узлах:')
                print(nodes_with_demand)
                #print('')
                #print('Спрос ПО ПРОГНОЗУ')
                #print(current_demand)
                #print('Спрос ПО ФАКТУ')
                #print(current_demand_fact)
                #print('')

            nodes_with_demand_n_depo = nodes_with_demand.copy()
            nodes_with_demand_n_depo.insert(0,'0')

            # Получить матрицу расстояний на текущий час
            with open(f'{matrix_data_path}/matrix_mon_{the_hour:02d}00_40_nodes.pkl', 'rb') as f:
                basic_matrix_df = pickle.load(f)
            mult_by = 4
            # Увеличить каждое значение времени перемещения в n раз (имитация перемещений на большие расстояния)
            for col in basic_matrix_df.columns:
                basic_matrix_df[col] = basic_matrix_df[col]*mult_by

            # Получить матрицу расстояний только для узлов потребления и депо
            time_matrix = basic_matrix_df[[int(n) for n in nodes_with_demand_n_depo]].loc[[int(n) for n in nodes_with_demand_n_depo]]
            time_matrix = time_matrix.values.tolist()
            index_mapping = {}
            for i in range(len(nodes_with_demand_n_depo)):
                index_mapping[nodes_with_demand_n_depo[i]] = i
            # Получить индекс узла депо
            depo_index = index_mapping[start_node]
            # Получить спрос (For or-tools demand should be in a list, where the node is identified by index)
            demand_list = [current_demand[n] for n in nodes_with_demand]
            # Insert start node with 0 demand
            demand_list.insert(depo_index, 0)
            # Определить, какие машины есть в депо (Если их нет, то ждём еще час до развозки, ессссно)
            vehicles_in_depo = [v for v in all_vehicles.keys() if
                                all_vehicles[v]['in_depo'] and all_vehicles[v]['returns_in'] == 0]

            if not vehicles_in_depo:
                if to_print:
                    print('Нет свободных машин')
            else:
                # Спланировать доставку
                routes+=1
                try:
                    vehicles_trips, covered_nodes_indexes, routing_time, nodes_by_routes = solve_cvrp(time_matrix, demand_list, depo_index,
                                                                             len(vehicles_in_depo), index_mapping,
                                                                             capacity, first, search_option, search_time, to_print)
                    vis_results.append(nodes_by_routes)
                    #my_plots.plot_routes(vis_results, routing_time, first, search_option, search_time)

                    # Обновить состояния машин, которые участвуют в доставке в этом часе и статистику по выездам
                    for v_ind in range(len(all_vehicles)):
                        if v_ind in vehicles_trips.keys():  # машина участвовала в планировании доставки
                            if vehicles_trips[v_ind] > 0:  # машина была отправлена на маршрут
                                if f'машина_{str(vehicles_in_depo[v_ind])}' not in vehicles_stats[f'день_{str(the_day)}'].keys():
                                    vehicles_stats[f'день_{str(the_day)}'][f'машина_{str(vehicles_in_depo[v_ind])}'] = vehicles_trips[v_ind]
                                else:
                                    vehicles_stats[f'день_{str(the_day)}'][f'машина_{str(vehicles_in_depo[v_ind])}'] += vehicles_trips[v_ind]
                                all_vehicles[vehicles_in_depo[v_ind]]['in_depo'] = False
                                all_vehicles[vehicles_in_depo[v_ind]]['returns_in'] = vehicles_trips[v_ind]
                    # Обновить запас в тех узлах, куда уехали машины (считаем, что если машины выехали, то груз будет доставлен)
                    # Исключить узел-депо из покрытых маршрутами узлов
                    covered_nodes_indexes = [cn for cn in covered_nodes_indexes if cn != depo_index]
                    # Обновить запас в узлах, покрытых маршрутами, начиная со следующего часа
                    for cn in covered_nodes_indexes:
                        covered_demand = demand_list[cn]
                        node_name = list(index_mapping.keys())[list(index_mapping.values()).index(cn)]
                        # Обновить запас в узлах ПО ПРОГНОЗУ
                        new_demand_schedule = [din_stock_predictions[node_name][ind] + covered_demand
                                               if ind > timer else din_stock_predictions[node_name][ind]
                                               for ind in range(len(din_stock_predictions[node_name]))]
                        din_stock_predictions[node_name] = new_demand_schedule
                        # Обновить запас в узлах ПО ФАКТУ
                        new_demand_schedule = [din_stock_fact[node_name][ind] + covered_demand
                                               if ind > timer else din_stock_fact[node_name][ind]
                                               for ind in range(len(din_stock_fact[node_name]))]
                        din_stock_fact[node_name] = new_demand_schedule
                    # Накопить время в пути
                    total_routing_time += routing_time

                except:  # задача VRP не нашла решения
                    if to_print:
                        print('Нет решения задачи CVRP при заданных условиях')
                    return 0,0,0,0

        # Если спроса в этом часе нет
        else:
            if to_print:
                print(f'\nВ течение {predict_ahead} часов запас не упадёт до критического значения. Доставка не планируется в этом часе')

        # Подготовка к следующему шагу таймера - у всех машин вне депо уменьшить на 60 минут время до возвращения и
        # при необходимости обновить статус
        for v in all_vehicles.keys():
            if all_vehicles[v]['returns_in'] > 0:
                returns_in = max(all_vehicles[v]['returns_in'] - 60 * timer_pace, 0)
                all_vehicles[v]['returns_in'] = returns_in
                if returns_in == 0:
                    all_vehicles[v]['in_depo'] = True

    end_time = datetime.now()
    duration = end_time - start_time
    print(f'\nВремя расчета: {duration}\n')

    # Рассчитать стоимость организации доставки по предложенному графику
    total_rent_cost = 0
    total_dist_cost = 0
    for day in vehicles_stats.keys():
        total_rent_cost += rent_cost[str(capacity)] * len(vehicles_stats[day])
        total_dist_cost += distance_cost[str(capacity)]*sum(vehicles_stats[day].values())

    return total_routing_time, routes, total_rent_cost, total_dist_cost


results = []

for const in [4,8,12,24]: # параметры таймера
    the_timer_pace = the_predict_ahead = const
    for the_vehicles in [5,8,10,12]: # количество машин
        for the_capacity in [4000,6000]: # грузоподъемность машин
            for the_warehouse_stock in [500,2000]: # объем склада узлов потребления
                for the_soft_limit in [0.8,1]: # "мягкое" пороговое значение
                    print(f'Машин: {the_vehicles}, грузоподъемностью: {the_capacity}, '
                          f'размер склада: {the_warehouse_stock}, мягкий лимит: {the_soft_limit}')

                    # Начальное значение запасов в узлах (если оно одинаковое для всех узлов)
                    # начинаем, когда запас во всех узлах соответствует мягкому лимиту
                    the_init_stock = int(the_warehouse_stock * the_soft_limit)
                    for first in ['PATH_CHEAPEST_ARC']:
                        for search_option in ['GUIDED_LOCAL_SEARCH']:
                            for search_time in [1]:
                                total_routing_time, routes, total_rent_cost, total_dist_cost = vrp_for_period(
                                    the_vehicles,
                                    the_capacity, the_warehouse_stock,
                                    the_init_stock, the_timer_pace, the_predict_ahead,
                                    the_soft_limit, first, search_option, search_time, to_print)
                                print(
                                    f'\nОбщее время в пути: {total_routing_time} мин ({round(total_routing_time / 60, 2)} ч)\n')

                                dict = {'Кол-во машин': the_vehicles, 'Грузоподъемность': the_capacity,
                                        'Размеры склада': the_warehouse_stock,
                                        'Частота таймера, ч': the_timer_pace,
                                        'Лимит склада': the_soft_limit,
                                        'Общее время в пути, ч': round(total_routing_time / 60, 2),
                                        'Выездов': routes,
                                        'Общая стоимость аренды': total_rent_cost,
                                        'Общая стоимость движения': total_dist_cost,
                                        'Общая стоимость': total_dist_cost + total_rent_cost,
                                        'First solution strategy': first,
                                        'Local search option': search_option,
                                        'Solve_search_time': search_time
                                        }
                                print('Выездов:', routes)
                                results.append(dict)

data = pd.DataFrame(results)
data.to_excel(f'datasets/wtf.xlsx')








