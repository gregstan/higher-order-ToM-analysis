"""Agent"""
import Players as play, Distributions as dst, pprint as pp, uuid, random, time
# import GameTree as gt

def istree(tree) -> bool:
    "Same as isinstance(tree, Tree) without importing GameTree.py"
    for attr in ["nplayers", "players", "idnum", "options"]:
        if not hasattr(tree, attr): return False
    return True


def agent(tree, player_tag: int | uuid.UUID) -> None:
    """
    Generates responses for all artificial agents on the platform.  

    This agent function takes an environment and a personality and generates responses.  The environment is a game
    tree and its personality is composed of agent parameters within a the 'players_dict' attribute attached to the 
    root of the tree.  agent() relies heavily on Node and Tree methods and attributes, all of which can be thought
    of as part of agent(): expected_payoffs, payoff_dimensions, utility, choice_probabilities, strategy_profiles, 
    and perspective_of_player.  
    
    Arguments:
        • tree: gt.Tree; The game tree to play upon.
        • player_tag: int | uuid; Determines which player this agent function computes 
            responses for, such as player 0, 1, or 2.  Must be a player already on the tree.
                
    Returns:
        • None but modifies the tree by adding responses.

    Raises:
        • TypeError: If tree is not a Tree instance from GameTree.py
        • ValueError: If the player number is not found on the tree
        • Exception: If the player number refers to a human player
    """

    if not istree(tree): 
        raise TypeError(f"Unsupported tree type {type(tree)}. Use type Tree from GameTree.py.")
    
    if isinstance(player_tag, int): 
        player_number = player_tag
        if not (0 <= player_number <= tree.nplayers - 1):
            raise ValueError(f"player_number {player_number} not found within tree.") 
    elif isinstance(player_tag, str):
        for player_index, player in enumerate(tree.players):
            if player_tag == player['uuid']:
                player_number = player_index
                break   
            elif player_index >= tree.nplayers - 1:
                raise ValueError(f"player uuid {player_tag} not found within tree.")    
    else: raise ValueError(f"Invalid player tag type {player_tag} type: {type(player_tag)}.")

    player_uuid = tree.players[player_number]['uuid']

    # if tree.players[player_number]['is_human']:
    if tree.players[player_number]['player_type'] != 'robot':
        raise Exception(f"Agent function cannot be called on a human player: {tree.players[player_number]}")

    this_node = tree.nodes[tree.current_nodeid]

    if this_node.isleaf() or this_node.choicetypeis("chance"): return None

    "By changing the tree perspective, the tree attributes will all be the attributes this player believes them to be."
    tree.perspective_of_player = player_number

    strategy_profile = tree.strategy_profiles[player_number]

    "Preventing agent from submitting redundant responses"
    response_array = this_node.choice if this_node.chooser[player_number] else this_node.prediction
    if response_array[player_number] is not None: 
        raise Exception(f"Artificial agent called to submit a redundant response")

    for player_index in range(tree.nplayers):
        if player_index != player_number:
            # if tree.players[player_index]["is_human"]:
            if tree.players[player_index]["player_type"] != "robot":
                error_message_end = f"before an artificial player {player_number}."
                if this_node.choice[player_index] is not None:
                    print(f"A human player {player_index} submitted a choice " + error_message_end)
                elif this_node.prediction[player_index] is not None:
                    print(f"A human player {player_index} submitted a prediction " + error_message_end)

    "Locating precomputed choice probabilities"
    children = tree.childrenof(nodeid=this_node.idnum)
    if this_node.choicetypeis("simultaneous") and this_node.predictor[player_number]:
        "Predictions of simultaneous choices are predictions about the final result."
        children = tree.childrenof(nodeid=this_node.idnum, depth=len(this_node.info_set))
        choice_probabilities_ = strategy_profile[children[0]:children[-1]+1]
        choice_probabilities = [round(probability / sum(choice_probabilities_), 4) if sum(\
            choice_probabilities_) > 0 else round(1/len(choice_probabilities_), 4) for probability in choice_probabilities_]
    else:
        children = tree.childrenof(nodeid=this_node.idnum)
        choice_probabilities = strategy_profile[children[0]:children[-1]+1]

    selected_nodeid = random.choices(population=children, weights=choice_probabilities, k=1)[0]
    selected_option_label = tree.olst[selected_nodeid]

    "Probabilistically selecting a reaction time based on rtime parameter"
    reaction_time_parameter = tree.players_dict[f"{player_number}rtime"]

    reaction_time_key_dn, reaction_time_key_up = dst.sample_distribution(
        distribution_type="normal", interval=[0, int(this_node.time[-1] * 2)], 
        coefficients=[reaction_time_parameter["mean"], reaction_time_parameter["var"]], 
        size=2)
    reaction_time_key_dn, reaction_time_key_up = 2.345, 3.456  #UNDO after testing

    reaction_time_key_dn = round(abs(reaction_time_key_dn), 3)
    reaction_time_key_up = round(reaction_time_key_dn + abs(reaction_time_key_up) / 3, 3)

    "Generating Response"
    # player = {"player_uuid": player_uuid, "player_num": player_number}
    # location = {"round_room_batch": tree.round_room_batch, "nodeid": tree.current_nodeid}
    # main = {"option": selected_option_label, "keypress": random.choice(["a", "d"]), 
    #         "rtimedn": reaction_time_key_dn, "rtimeup": reaction_time_key_up, "timestamp": time.time()}
    # response = {"player": player, "location": location, "main": main}

    metadata = {"player_uuid": player_uuid, "player_num": player_number, 
                "round_room_batch": tree.round_room_batch, "nodeid": tree.current_nodeid}
    data = {"option": selected_option_label, "keypress": random.choice(["a", "d"]), 
            "rtimedn": reaction_time_key_dn, "rtimeup": reaction_time_key_up, "timestamp": time.time()}
    response = {"metadata": metadata, "data": data}

    "Applying response to tree"
    # tree.response_to_tree(response=response, seconds_since_round_started=response["main"]["rtimeup"])
    tree.response_to_tree(response=response, seconds_since_round_started=response["data"]["rtimeup"])

    "Resetting the perspective to be 'objective'"
    tree.perspective_of_player = None



