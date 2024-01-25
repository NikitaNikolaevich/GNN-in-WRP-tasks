from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model(time_matrix, demand, depo_index, num_vehicles, capacity:int):
    data = {}
    # Матрица расстояний получена на основании сетевого графа Василеостровского района (с округлением до 2го знака)
    # ! Сейчас это время в минутах!
    data["distance_matrix"] = time_matrix
    data["num_vehicles"] = num_vehicles
    data["depot"] = depo_index
    data["demands"] = demand
    data["vehicle_capacities"] = [capacity for n in range(num_vehicles)]
    return data

def print_solution(data, manager, routing, solution, index_mapping, to_print):
    vehicles_trips = {}
    covered_nodes = []
    nodes_by_routes = []
    """Prints solution on console."""
    # print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        # Для визуализации маршрутов
        nodes_by_car = []
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            # Получить номер узла в графе по его индексу
            node_name = list(index_mapping.keys())[list(index_mapping.values()).index(node_index)]
            plan_output += f"Node '{node_name}' Delivered({route_load}) -> "
            nodes_by_car.append(node_name)
            covered_nodes.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        nodes_by_routes.append(nodes_by_car)
        # Учесть время на разгрузку (5 минут для всех узлов)
        route_distance += (len(nodes_by_car)-1)*5

        plan_output += f" Node '{list(index_mapping.keys())[list(index_mapping.values()).index(manager.IndexToNode(index))]}' " \
                       f"Delivered({route_load})\n"
        plan_output += f"Time of the route: {route_distance}min\n"
        plan_output += f"Load of the route: {route_load}\n"
        if to_print:
            print(plan_output)
        total_distance += route_distance
        total_load += route_load
        # Output about vehicle trips
        vehicles_trips[vehicle_id] = route_distance
    if to_print:
        print(f"Total time of all routes: {total_distance}min")
        print(f"Total load of all routes: {total_load}")
    return vehicles_trips, covered_nodes, total_distance, nodes_by_routes

def solve_cvrp(time_matrix, demand, depo_index, num_vehicles, index_mapping, capacity:int, first, search_option, search_time, to_print):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model(time_matrix,demand,depo_index, num_vehicles, capacity)
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    '''
    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        300,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(300)
    '''

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    if first == 'PATH_CHEAPEST_ARC':
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    elif first == 'PATH_MOST_CONSTRAINED_ARC':
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC)
    elif first == 'GLOBAL_CHEAPEST_ARC':
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC)
    elif first == 'FIRST_UNBOUND_MIN_VALUE':
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE)
    else:
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    if search_option == 'GUIDED_LOCAL_SEARCH':
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    if search_option == 'GREEDY_DESCENT':
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT)
    if search_option == 'SIMULATED_ANNEALING':
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
    if search_option == 'GENERIC_TABU_SEARCH':
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH)
    else:
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    search_parameters.time_limit.FromSeconds(int(search_time))

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        vehicles_trips, covered_nodes, total_distance, nodes_by_routes = print_solution(data, manager, routing, solution, index_mapping, to_print)
        return vehicles_trips, covered_nodes, total_distance, nodes_by_routes
    else:
        if to_print:
            print("No solution found !")
        return None