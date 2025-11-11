"""Players"""
import pprint as pp, copy, uuid, datetime, time, random, json, os
from cryptography.fernet import Fernet
from collections import defaultdict

player_type_: str = 'robot' or 'guest' or 'participant' or 'researcher' or 'developer'
player_types = ('robot', 'guest', 'participant', 'researcher', 'developer')

def experimentid(start_date: str = None, experimenter_uuid: str = None, simulated: bool = False):
    """
    Generates an experiment id based on the experiment start date and th experimenter's uuid.
    
    Arguments:
        • start_date: str (optional); Formatting: 'YYYY-MM-DD-HH-MM'. Example: '2024-02-05-14-00' 
        • experimenter_uuid: str | None; such as d309cac2-b2d0-49f6-b5eb-5b1765384a22.  If None, 
            this will generate a random uuid. 
        • simulated: bool; If True, this will add 'SIM' to the experiment id to clarify that 
            this is not a real experiment.
    
    Returns:
        • str: A unique experiment id
    """
    
    if start_date is None: start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    if experimenter_uuid is None: experimenter_uuid = uuid.uuid4()

    sim = "SIM" if simulated else ""

    return f"{sim}EXPID-{experimenter_uuid}-{start_date}"


class ParticipantsData(dict):
    """
    Stores all player information needed to run an experiment, including avatar appearances, 
    sids, and game histories.  Players can be identified by uuids, user numbers, or sids.
    User names and email addresses are encrypted to preserve anonymity.
    """
    def __init__(self) -> None:
        """
        Constructs all necessary attributes for the ParticipantsData object, including setting the number of players 
        to zero, initializing the experiments dictionary, and setting up the encryption key and cipher suite.

        Example:
        >>> participants_data = ParticipantsData()
        """
        self.nplayers = 0
        self.experiments = {}
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    def __getitem__(self, key) -> 'Player':
        """
        Special method to retrieve a player instance when indexing into ParticipantsData with a key.
        
        It overrides the dict's __getitem__ method and handles different types of keys (UUID, user_number, or sid).
        If the key is a slice, it returns a list of players; otherwise, it converts the key to a UUID and retrieves the player.

        Example:
        >>> player = participants_data[0]
        >>> print(player)
        """
        if isinstance(key, slice):
            return [self[key_] for key_ in self.user_ids[key]]
        key = self.key_to_uuid(key=key)
        return super().__getitem__(key)

    def __getattr__(self, name) -> object:
        """
        Special method to dynamically handle attribute access that aren't explicitly defined.

        It handles four attributes: 'user_numbers', 'user_ids', 'web_sockets_to_uuids', and 'user_numbers_to_uuids'.
        For any other attribute, it raises an AttributeError.

        Example:
        >>> user_numbers = participants_data.user_numbers
        """
        if name == 'user_numbers':
            "List of user numbers from 0 to nplayers"
            return list(range(len(self)))
        if name == 'user_ids':
            "Sorted list of uuids"
            return sorted(list(self.keys()))
        if name == 'web_sockets_to_uuids':
            "Dictionary mapping sids to uuids, generated on the spot to ensure it is not out of sync."
            return {val["sid"]: key for key, val in self.items()}
        if name == 'user_numbers_to_uuids':
            "Dictionary mapping user numbers to uuids, generated on the spot to ensure it is not out of sync."
            return {val["user_number"]: key for key, val in self.items()}
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setitem__(self, key, val) -> None:
        super(ParticipantsData, self).__setitem__(key, val)
        """
        Special method to add a new player to ParticipantsData and updates all players' user numbers.

        It overrides the dict's __setitem__ method, increments the number of players and updates 
        the user number for all players who have a greater user number than the newly added player.

        Example:
        >>> participants_data[0] = Player()
        """
        self.nplayers += 1
        uuid_ = self.key_to_uuid(key=key)
        unum_ = self[uuid_].user_number
        for player in self.values():
            if player.user_number > unum_:
                player.user_number += 1        

    def __delitem__(self, key) -> None:
        """
        Special method to remove a player from ParticipantsData and updates all players' user numbers.

        It decrements the number of players and updates the user number for all players who have 
        a greater user number than the player being removed.

        Example:
        >>> del participants_data[0]
        """
        self.nplayers -= 1
        uuid_ = self.key_to_uuid(key=key)
        unum_ = self[uuid_].user_number
        for player in self.values():
            if player.user_number > unum_:
                player.user_number -= 1  
        return super().__delitem__(uuid_)

    def sid_to_uuid(self, sid: str) -> uuid.UUID:
        """
        Converts a sid string to the corresponding UUID.

        Example:
        >>> uuid = participants_data.sid_to_uuid('wss://moralitygame.example0.com/')
        """
        if not isinstance(sid, str):
            raise ValueError("sids must be strings.")
        return self.web_sockets_to_uuids[sid]

    def unumber_to_uuid(self, unumber: int) -> uuid.UUID:
        """
        Converts a user number to the corresponding UUID.

        Example:
        >>> uuid = participants_data.unumber_to_uuid(0)
        """
        if not isinstance(unumber, int):
            raise ValueError("User numbers must be integers.")
        if unumber > self.nplayers:
            raise ValueError(f"Input user number: {unumber} > maximum user number: {self.nplayers}!")
        return self.user_numbers_to_uuids[unumber]
    
    def key_to_uuid(self, key: str | int) -> uuid.UUID:
        """
        Converts any type of key (user number, sid, or UUID) to the corresponding UUID.

        Example:
        >>> uuid = participants_data.key_to_uuid(0)
        """
        if isinstance(key, int):
            return self.unumber_to_uuid(unumber=key)
        elif isinstance(key, str):
            if key.startswith("wss://"):
                return self.sid_to_uuid(sid=key) 
            else: return key   
        else: raise ValueError(f"Unsupported key type {type(key)}")    

    def __repr__(self) -> None:
        """
        Prints ParticipantsData in a way that is easiest to visually parse.
        """
        players_repr = "\n\n".join([f"{uuid}: {player_repr}" for uuid, player_repr in self.items()])
        return f"{{\n{players_repr}\n}}"

    class Player(dict):
        """
        Dictionary storing all information relevant to a human or artificial player.

        Instances of Player are stored at the root of each game tree in the 'players' Tree attribute.

        Attributes:
            • players_data: ParticipantsData; The ParticipantsData dictionary that this Player belongs to
            • user_number: int; An integer used to identify the player that is more convenient to work with
            • uuid: uuid.UUID; Universal Unique Identifier, a 128-bit string used to identify the player.
            • player_type: 'robot' | 'guest' | 'participant' | 'researcher' | 'developer'; The 
                category of the agent, which determines their available actions on the website.            
            • sid: str; The sid id for the player, like wss://moralitygame.example0.com/
            • username: str; Encrypted user name of player, like gregstan
            • email_address: str; Encrypted email address of player
            • avatar: dict[str: str]; Avatar shape, color, and texture
            • cummulative_payoffs: int; The number of payoffs already won.
            • experiments: list[str]; List of experiments participanted in
            • tutorials: list[str]; List of tutorial modules passed
            • history: list[str]; List of titles of trees played
        """
        def __init__(self, players_data: 'ParticipantsData', username: str, email_address: str, uuid_: str = None, sid_: str = None,
                     player_type: str = 'robot', avatar_color: tuple[int, int, int, float] = None, avatar_shape: str = None):
            self.players_data: ParticipantsData = players_data
            self.user_number: int = len(players_data)
            self.uuid: uuid.UUID = str(uuid.uuid4()) if uuid_ is None else uuid_
            self.sid: uuid.UUID = str(uuid.uuid4()) if sid_ is None else sid_
            self.username: str = self.encrypt_data(username)
            self.email_address: str = self.encrypt_data(email_address)
            self.avatar: dict[str: str] = self.avatar_look(hsla_color=avatar_color, shape=avatar_shape)
            self.player_type: str = player_type if player_type in player_types else 'robot'
            self.cumulative_payoffs = 0
            self.experiments: list[str] = []
            self.tutorials: list[str] = []
            self.history: list[str] = []
            
            self.attrs_lst_player = ['user_number', 'uuid', 'sid', 'avatar', 
                                     'cumulative_payoffs', 'player_type']
            for attr in self.attrs_lst_player: self[f'{attr}'] = eval(f'self.{attr}')

        def __getattr__(self, name):
            """
            Automatically decrypts user names and email addresses while preserving encryption of stored data.
            """
            if name == 'decrypted_username':
                return self.decrypt_data(self.username)
            if name == 'decrypted_email_address':
                return self.decrypt_data(self.email_address)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def __setitem__(self, key, val) -> None:
            """
            Allows attributes to be set and accessed like dictionary keys.
            """
            super(ParticipantsData.Player, self).__setitem__(key, val)
            try: setattr(self, f"_{key}", val)
            except: pass

        def __setattr__(self, name, value) -> None:
            super(ParticipantsData.Player, self).__setattr__(name, value)
            if name.lstrip('_') in self: super(ParticipantsData.Player, self).__setitem__(name.lstrip('_'), value)

        def __repr__(self) -> str:
            """
            Prints Player in a way that is easiest to visually parse.
            """
            repr_dict = {key: self[key] for key in self.attrs_lst_player}
            return "{\n" + pp.pformat(repr_dict, indent=4, sort_dicts=False).replace("\n", "\n    ") + "\n}"

        @property
        def username(self) -> str:
            return self._username

        @username.setter
        def username(self, value) -> None: 
            """
            Automatically encrypts user names.
            """
            if not self.isencrypted(data=value): 
                self._username = self.encrypt_data(value)
            else: self._username = value

        @property
        def email_address(self) -> str:
            return self._email_address

        @email_address.setter
        def email_address(self, value) -> None:  
            """
            Automatically encrypts email addresses.
            """
            if not self.isencrypted(data=value): 
                self._email_address = self.encrypt_data(value)
            else: self._email_address = value

        @property
        def user_number(self) -> int:
            return self._user_number

        @user_number.setter
        def user_number(self, value) -> None:     
            """
            Updating default attributes when the user number changes.
            """
            self._user_number = value 
            name_prefix = "username"
            email_suffix = "@example.com"
            socket_prefix = "wss://moralitygame.example"
            if hasattr(self, "username"):
                if self.sid.startswith(socket_prefix):
                    self.sid = f"wss://moralitygame.example{value}.com/"
                if self.decrypted_username.startswith(name_prefix):
                    self.username = f"username_{value}" 
                if self.decrypted_email_address.endswith(email_suffix):
                    self.email_address = f"email{value}@example.com"

        def isencrypted(self, data) -> bool:
            """
            Determines if data has already been encrypted.
            """
            min_encryption_len = 30
            if "@" in data or len(data) < min_encryption_len: 
                return False
            else: return True

        def encrypt_data(self, data) -> str:
            """
            Encrypts data if not already encrypted.
            """
            if self.isencrypted(data):
                raise ValueError(f"Data is already encrypted {data}")
            return self.players_data.cipher_suite.encrypt(data.encode()).decode()

        def decrypt_data(self, encrypted_data) -> str:
            """
            Decrypts data if not already decrypted.
            """
            if not self.isencrypted(encrypted_data):
                raise ValueError(f"Data is already decrypted {encrypted_data}")
            return self.players_data.cipher_suite.decrypt(encrypted_data.encode()).decode()

        def _hsla(self, hue: int, satur: int = 100, light: int = 50, alpha: float = 0.6, huemax_: int = 360) -> str:
            """
            Returns an hsla string for determining colors.  This is for convenience.
            """ 
            return f"hsla({int((hue*360/(huemax_+1))%360)}, {int(satur)}%, {int(light)}%, {round(alpha, 2)})"

        def _get_avatar_looks(self, file_path: str = "./", file_name: str = "avatar_looks.json") -> dict[str: str]:
            """
            Retrieves avatar appearance information from json file.
            """
            full_path = os.path.join(file_path, file_name)
            if os.path.exists(full_path):
                with open(full_path, "r") as f:
                    avatar_look = json.load(f)
                return avatar_look
            else: raise Exception(f"Cannot find {file_name} in {file_path}.")

        def avatar_look(self, hsla_color: tuple[int, int, int, float] = None, 
                        shape: str = None, texture: str = None) -> dict[str: str]:
            """
            Assigns a avatar a color, shape, and texture.  
            
            avatar_looks.json contains a list of supported shapes and textures.

            Any arguments set to None, will be assigned randomly.

            Arguments:
                • hsla_color: tuple[int, int, int, float]; 
                    Accepted Ranges:
                    - Hue: int; 0 - 359
                    - Satur: int; 50 - 100
                    - Light: int; 35 - 65
                    - Alpha: float; 1.0
                • shape: str; Name of shape of avatar
                • png file name of texture of avatar
            """
            if hsla_color is None:
                hue = random.choice(list(range(0, 359)))
                satur = random.choice(list(range(50, 100)))
                light = random.choice(list(range(35, 65)))
                hsla_color = (hue, satur, light, 1.0)
            elif isinstance(hsla_color, str) and 'hsla' in hsla_color:
                for char in ['hsla(', '%', ')']:
                    hsla_color = hsla_color.replace(char, '')
                hsla_vals = hsla_color.split(', ')
                hsla_color = (int(hsla_vals[0]), int(hsla_vals[1]), int(hsla_vals[2]), round(float(hsla_vals[3]), 2))

            if not isinstance(hsla_color, (tuple, list)) or len(hsla_color) != 4:
                raise ValueError("hsla_color must be a tuple of four numbers.")
            if not isinstance(hsla_color[0], int):
                raise ValueError("Hue must be an integer between 0 and 359.")
            if not (50 <= hsla_color[1] <= 100): 
                raise ValueError("Saturation must be an integer between 50 and 100.")
            if not (35 <= hsla_color[2] <= 65): 
                raise ValueError("Light must be an integer between 35 and 65.")
            if not hsla_color[3] == 1.0: hsla_color[3] = 1.0

            hue, satur, light, alpha = hsla_color
            hsla_color = self._hsla(hue=hue, satur=satur, light=light, alpha=alpha)

            self.avatar_looks = self._get_avatar_looks()
            avatar_shapes = self.avatar_looks["shapes"]
            avatar_textures = self.avatar_looks["textures"]

            if isinstance(shape, str):
                if shape not in avatar_shapes:
                    raise ValueError(f"shape {shape} not in supported shapes: {avatar_shapes}.")
            elif shape is None:
                shape = random.choice(avatar_shapes)

            if isinstance(texture, str):
                if texture not in avatar_textures:
                    raise ValueError(f"texture {texture} not in supported textures: {avatar_textures}.")
            elif texture is None:
                texture = random.choice(avatar_textures)

            return {"shape": shape, "color": hsla_color, "texture": texture}

    def create_player(self, username: str = None, email_address: str = None, uuid_: str = None,
                      sid_: str = None, player_type: str = 'robot', avatar_shape: str = None, avatar_color: tuple[int, int, int, float] = None) -> None:
        """
        Creates a new player.  Uses defaults if arguments are None.
        """
        if username is None: username = f"username_{self.nplayers}"
        if email_address is None: email_address = f"email{self.nplayers}@example.com"
        new_player = self.Player(players_data=self, username=username, email_address=email_address, uuid_=uuid_, 
                                 sid_=sid_, player_type=player_type, avatar_shape=avatar_shape, avatar_color=avatar_color)
        self[new_player["uuid"]] = new_player

    def to_experiment(self, uuids: set[str], experimenter_uuid: str = None, start_date: str = None) -> None:
        """
        Creates a subdictionary with only the participating players.
        """
        if experimenter_uuid is None: experimenter_uuid = uuid.uuid4()
        if start_date is None: start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

        expid = f"EXPID-{experimenter_uuid}-{start_date}"

        for uuid_ in uuids:
            if isinstance(uuid_, int):
                uuid_ = self.unumber_to_uuid(unumber=uuid_)
            if uuid_ not in self:
                raise ValueError(f"to_experiment expects all users to already be saved. Player not saved: {uuid_}")
        self.experiments[expid] = ExperimentParticipantsData(all_players=self, uuids=uuids)


class ExperimentParticipantsData(ParticipantsData):
    """
    A subset of the main ParticipantsData object that contains information 
    pertaining to players currently participating in an experiment.
    """
    def __init__(self, all_players, uuids: set[str]):
        self.players_dict = all_players
        self.uuids = sorted(list(set([uuid_ if isinstance(uuid_, str) else self.players_dict.user_numbers_to_uuids[uuid_] for uuid_ in uuids])))
        self.sids_to_uuids = {self.players_dict[uuid_]["sid"]: uuid_ for uuid_ in self.uuids}

    def key_to_uuid(self, key):
        if isinstance(key, int):
            uuids = sorted(self.uuids)
            if 0 <= key < len(uuids):
                return uuids[key]
            else: raise KeyError("User number out of range")
        elif isinstance(key, str):
            if key in self.sids_to_uuids:
                return self.sids_to_uuids[key]
            elif key in self.uuids:
                return key
            else: raise KeyError("Invalid sid or UUID")
        else: raise KeyError("Unsupported key type")

    def __getitem__(self, key):
        uuid = self.key_to_uuid(key)
        if uuid not in self.uuids:
            raise KeyError("UUID not part of this experiment")
        return self.players_dict[uuid]

    def __setitem__(self, key, value):
        uuid = self.key_to_uuid(key)
        self.players_dict[uuid] = value
        self[uuid] = value

    def __iter__(self):
        for uuid_ in sorted(self.uuids):
            yield uuid_

    def __len__(self):
        return len(self.uuids)


class ContextualDict(dict):
    """
    A dictionary that allows items to be defined by referencing other
    items, instance variables of an external object, or global variables,
    even if they have not yet been defined.

    Example Uses: 

    To reference another dictionary item:
        sdict = ContextualDict()
        sdict['key1'] = "{key2} * 2"
        sdict['key2'] = 5
        sdict['key1']
        >>> 10

    To reference an instance variable of an external object:
        tree = Tree(...)
        sdict = ContextualDict(external_context=tree)
        sdict['key1'] = "{current_nodeid} * 2"
        tree.current_nodeid = 5
        sdict['key1']
        >>> 10

    To reference a global variable:
        sdict = ContextualDict()
        sdict['key1'] = "{key2} * 2"
        key2 = 5
        sdict['key1']
        >>> 10

    Note that the variable inside the braces must be defined by the time
    the ContextualDict item is accessed, not necessarily when it is set.

    To perform more complex operations:

        Write a conditional within the string:
            sdict = ContextualDict(external_context=tree)
            sdict['0b:1v0'] = "0.9 if {current_nodeid} <= 2 else 0.96 if {current_nodeid} == 3 else 0.2"
            tree.current_nodeid = 4
            sdict['key1']
            >>> 0.2

        Or include a function within the string:
            def multiply_by_2(number: int) -> int: return number * 2
            sdict['key1'] = "multiply_by_2(number={key2}) * 2"
            sdict['key2'] = 5
            sdict['key1']
            >>> 10
    """
    def __init__(self, external_context: object = None):
        super().__init__()
        self.external_context = external_context
        self.previous_values = {}
        self.external_refs = {}
        self.defaults_ = {}
        
    def _parse_value_string(self, value) -> object | None:
        "Replace keys with their values in the expression string."
        while "{" in value:
            start = value.index("{")
            end = value.index("}") + 1
            key = value[start+1:end-1]
            try:
                if key in self.external_refs:
                    val = self._parse_value_string(self.external_refs[key]) 
                    value = value[:start] + str(val) + value[end:]
                elif key in self:
                    value = value[:start] + str(self[key]) + value[end:]
                elif self.external_context and hasattr(self.external_context, key):
                    value = value[:start] + str(getattr(self.external_context, key)) + value[end:]
                elif key in globals():
                    value = value[:start] + str(globals()[key]) + value[end:]
                else:
                    return None
            except KeyError:
                return None
        return eval(value)

    def __setitem__(self, key, value) -> None:
        if isinstance(value, str) and key in value and key in self.previous_values:
            self.modify(key=key, expression=value)
        elif isinstance(value, str) and "{" in value:
            self.external_refs[key] = value
        else: 
            super().__setitem__(key, value)

    def __getitem__(self, key) -> object | None:
        if key in self.external_refs:
            value = self._parse_value_string(self.external_refs[key])
            if value is not None:
                "save the value for future use"
                self.previous_values[key] = copy.deepcopy(value) 
                return value
            else: 
                return None
        elif key in self.defaults_:
            return self.defaults_[key]
        elif key in self:
            return super().__getitem__(key)
        else: 
            return None

    def modify(self, key: str, expression: str) -> None:
        """
        Modify the value of a key with a string expression that is evaluated at runtime.
        The expression can reference the current value of the key, other keys, instance 
        variables of the external object, or global variables.

        Example Use (Manually):
        # Assuming that '0b:1v0' has already been saved
        sdict.modify('0b:1v0', "0b:1v0 + 0.2 if {current_nodeid} <= 4 else 0b:1v0 - 0.1")

        However, __setitem__ automatically calls modify if key is a 
        substring of value, so there is no need to manually use modify

        Example Use (Automatically):
        # Assuming that '0b:1v0' has already been saved
        sdict['0b:1v0'] = "{0b:1v0} + 0.2 if {current_nodeid} <= 4 else {0b:1v0} - 0.1"
        """

        "Replacing key substring with previous value with or without brackets around it."
        expression = expression.replace("{" + key + "}", f"{self.previous_values[key]}")
        expression = expression.replace(key, f"{self.previous_values[key]}")

        "Setting the modified item"
        self[key] = expression

    def __repr__(self):
        return repr(dict(self))


class HistoricalDict(dict):
    """
    A smart dictionary that remembers the history of previous entries + timestamps.
    """
    time_type = int | float | str

    def __init__(self, zero_time: int | float = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history = defaultdict(list)
        self.zero_time = time.time() if zero_time is None else zero_time

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if key in self._history:
            return self._history[key][-1][0]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        """
        Subtracts zero_time from each timestamp so that each time stamp shows time since timer began.
        """
        self._history[key].append((super().__getitem__(key), time.time()))

    def _convert_time(self, time_: time_type, fmt: str) -> time_type:
        """
        Converts time between integer, float, and string data types.
        """
        if fmt == "int": return int(time_)
        elif fmt == "float": return time_
        elif fmt == "str": return time.strftime("%H:%M:%S", time.gmtime(time_))
        else: raise ValueError("Invalid format specified. Use 'int', 'float', or 'str'.")

    def history(self, key: str, with_time_stamps: bool = False, time_fmt: str = "float") -> list[object] | list[tuple[object, time_type]]:
        """
        View the history of entries for key.  If with_time_stamps, returns time stamps for each entry.
        """
        if with_time_stamps:
            "Returns list of tuples, like [(value0, timestamp0), (value1, timestamp1), (value2, timestamp2),...]"
            return [(event[0], self._convert_time(time_ = event[-1] - self.zero_time, fmt=time_fmt)) for event in self._history[key]]
        "Returns list of values like [value0, value1, value2,...]"
        return [entry[0] for entry in self._history[key]]

    def delete_memory(self, key: str, index: int | None = None, timestamp: float | None = None) -> None:
        """
        Wipes memory for a key after a specific index or after a specific 
        timestamp.  If index and timestame are None, then it wipes all memory.
        """
        if key not in self._history:
            raise KeyError(f"No history found for key '{key}'")
        if index is not None:
            self._history[key] = self._history[key][:index]
        elif timestamp is not None:
            self._history[key] = [entry for entry in self._history[key] if entry[1] - self.zero_time < timestamp]
        else: del self._history[key]

    def __repr__(self) -> str:
        return repr({key: self[key] for key in self})

"Helps explain the meanings of key-value pairs in PlayerParameters via the totext method."
param_meanings = {
    'depth': 'can reason at a depth of', 
    'risk':  'has a risk aversion level of', 
    'loss':  'has a loss aversion level of', 
    'rtime': 'has an average reaction time of',
    'dtime': 'discounts the future at a rate of',
    'yvne':  'dislikes advantageous inequality at',
    'envy':  'dislikes disadvantageous inequality at'
    }

class PlayerParameters(dict):
    """
    PlayerParameters is a dictionary that stores human and artificial player parameters, including risk
    aversion, loss aversion, reasoning depth, average reaction time, and the strengths of social connections.
    These parameters determine the choices and predictions of bots when entered into agent functions.  Also, 
    bots infer the parameters of other players and use this dictionary as a memory bank when doing so.  To
    mitigate a combinatorial explosion of parameters, this dictionary allows one to retrieve items that have
    not yet been saved by using heuristics and default values.  

    Key Formatting Examples: '1b:2b:3b:4loss' means that, 'Player 1 believes that player 2 believes that player 
    3 believes that player 4 has a loss aversion level of...'.  '4risk' means that, 'Player 4 has a risk aversion
    level of...'.  '2v3' means that, 'Player 2 values player 3 at a level of...'.  

    Heuristics: Unless otherwise specified: (1) Higher-order beliefs resemble lower-order beliefs, and (2) beliefs
    are true.  So this dictionary will always return saved values but otherwise will use these heuristics to return
    lower-order beliefs, true values, or default values.
    """
    number_type = int | float

    def __init__(self, players: ParticipantsData | int, external_context: object = None, *args, **kwargs):
        self.param_types = ['values', 'risk', 'loss', 'yvne', 'envy', 'depth', 'rtime', 'dtime']
        
        self._defaults = {
            'iVi':   {"mean": 1.0, "var": 0.2}, 'iVj':   {"mean": .32, "var": 0.4}, 
            'risk':  {"mean": 0.8, "var": 0.1}, 'loss':  {"mean": 1.2, "var": 0.1},  
            'yvne':  {"mean": 0.2, "var": 0.2}, 'envy':  {"mean": 0.4, "var": 0.3}, 
            'depth': {"mean": 6.0, "var": 2.0}, 'rtime': {"mean": 2.5, "var": 4.0}, 
            'dtime': {"mean": 1.0, "var": 0.0}}
        self.defaults_ = self._defaults
        self.contextual = ContextualDict(external_context=external_context)

        if isinstance(players, int):
            self.players_dict = ParticipantsData()
            for pdx in range(players):
                self.players_dict.create_player()
        elif isinstance(players, ParticipantsData | ExperimentParticipantsData):
            self.players_dict = players
        else: raise ValueError("players must be a ParticipantsData dict or the number of players to create.")

    def update_defaults(self, defaults_dict: dict) -> None:
        """
        Updates the default values in the PlayerParameters
        """
        for val in defaults_dict.values():
            if not isinstance(val, dict):
                raise ValueError(f"{type(val)} not supported - defaults_dict must contain dictionaries!")
            if not "mean" in val or not "var" in val:
                raise ValueError("All defaults must contain a mean and a variance.")        
            if not isinstance(val["mean"], (int | float)) or not isinstance(val["var"], (int | float)):
                raise ValueError("mean and var must be numeric.")
        self._defaults.update(defaults_dict)

    def __getattr__(self, name):
        if name == "keys_":
            "Sorted list of dictionary keys"
            return sorted(list(self.keys()), key=lambda x: (len(x), x))
        if name == 'user_numbers':
            "List of user numbers from 0 to nplayers"
            return list(range(len(self.players_dict)))
        if name == 'uuids':
            "Sorted list of user ids"
            return sorted(list(self.players_dict.user_ids))

    def key_to_list(self, key: str) -> list[int | str]:
        """
        Parses the key and returns a list of uuids and parameters.
        
        Examples:
            • 6v3 -> ['6', 'values', '3']
            • 9depth -> ['9', 'depth', '']
            • 1b:3risk -> ['1', '3', 'risk', '']
            • 3b:4b:5b:8loss -> ['3', '4', '5', '8', 'loss', '']
            • 2b:6b:6v2 -> ['2', '6', '6', 'values', '2']
        """
        if "b:" in key:
            beliefs = key.split("b:")
        else: beliefs = key.split("believes:")
        belief_lst, remainder = beliefs[:-1], beliefs[-1]

        if "values" in remainder:
            rsplit = remainder.split("values")
            return belief_lst + [rsplit[0]] + ["values"] + [rsplit[-1]]
        if len(remainder) < 33 and "v" in remainder and \
            not any(remainder.endswith(param) for param in self.param_types[1:]):
            rsplit = remainder.split("v")
            return belief_lst + [rsplit[0]] + ["values"] + [rsplit[-1]]
        for param in self.param_types[1:]:
            if key.endswith(param):
                return belief_lst + [remainder.replace(param, "")] + [param, ""]
        else: return []

    def key_properly_formatted(self, key: str) -> bool:
        """
        Determines if a key is properly formatted before it is used to index a value in the dictionary
        """
        if not isinstance(key, str): return False
        if not self.key_to_list(key): return False
        return True

    def val_properly_formatted(self, value: dict[str: float]) -> bool:
        """
        Determines if a value is properly formatted before it is entered into the dictionary
        """
        if not isinstance(value, dict): return False
        if "mean" not in value or "var" not in value: return False
        if not isinstance(value["mean"], (int, float)): return False
        if not isinstance(value["var"], (int, float)): return False
        return True

    def tolongform(self, key: str) -> str:
        """
        Converts the key into a longer format, which includes full uuids, not user numbers, and 'believes', not 'b:'.
        """
        key = key.replace("@n", "")
        key = key.replace("~", "")
        keylst = self.key_to_list(key)
        try:
            beginning = "".join([f"{self.uuids[int(ele)]}believes:" if len(ele) < 32 \
                and ele.isdigit() else f"{ele}believes:" if ele != "?" else "?" for ele in keylst[:-3]])
            ending = "".join([self.uuids[int(ele)] if len(ele) < 32 \
                and ele.isdigit() else ele for ele in keylst[-3:]])
            return beginning + ending
        except IndexError:
            raise Exception("Not enough players have been added to this dictionary.")
        
    def toshortform(self, key: str) -> str:
        """
        Converts the key into a shorter format, which includes user numbers, not full uuids, and 'b:', not 'believes'.
        """
        keylst = self.key_to_list(key)
        beginning = "".join([f"{ele}b:" if len(ele) < 32 and ele.isdigit() else \
            f"{self.uuids.index(ele)}b:" if ele != "?" else "?" for ele in keylst[:-3]])
        ending = keylst[-3] if len(keylst[-3]) < 32 and \
            keylst[-3].isdigit else str(self.uuids.index(keylst[-3]))
        if keylst[-1] != "":
            tail = keylst[-1] if len(keylst[-1]) < 32 and \
                keylst[-1].isdigit else str(self.uuids.index(keylst[-1]))
            ending += f"v{tail}"
        else: ending += keylst[-2]    
        return beginning + ending    

    def isuuid(self, key: str) -> bool:
        """
        Returns if the key is an instance of a 'uuid'.
        """
        if isinstance(key, uuid.UUID): return True
        if "-" in key:
            if [len(subuuid) for subuuid in key.split("-")] == [8, 4, 4, 4, 12]:
                return True
        return False

    def _parameter_in_key(self, key: str) -> str:
        """
        Returns the parameter within the key, such as 'values' or 'envy'
        """
        return self.key_to_list(key)[-2]

    def itterkey(self, key: str) -> list[str]:
        """
        If '?' in key, this means all possible players, so if '1b:2v?' == 0.9, 
        then this means that player 1 believes that player 2 values everyone else 
        at 0.9.  This method returns a list of all the permutations of keys.  If 
        '?' not in key, then this method returns [key].
        """
        key = self.tolongform(key)

        if "?" not in key: return [key]

        keylst = self.key_to_list(key)

        nqs = [qkey for qkey in keylst if qkey == "?"]
        if len(nqs) >= 2: raise ValueError(f"Instead of writing a double '?', just update default iVj.")

        return [key.replace("?", self.uuids[idx]) for idx in range(len(self.uuids))]

    def reduce_belief_level(self, key: str) -> str:
        """
        Reduces the level of the belief within the belief key prefix.

        Assuming that higher-order beliefs tend to resemble lower-order 
        beliefs, this takes a key like '1b:2b:1b:...' and trims it to '1b:...'.
        """
        if "?" in key:
            raise ValueError("'?' in key. First pass key though itterkey.")

        if "b:" not in key and "believes:" not in key:
            return key

        keylst = self.key_to_list(key)
        believer = keylst[0]
        truth = keylst[-3:]
        for bdx, player in enumerate(keylst[:-3]):
            if bdx > 0 and player == believer:
                bkey = ""
                for belief in keylst[bdx:-3]:
                    if belief.isdigit():
                        bkey += f"{belief}b:"
                    else: bkey += f"{belief}believes:"
                return bkey + "".join(truth)
        return "".join(truth)

    def __getitem__(self, key: str) -> dict[str: float]:
        """
        If the key is in the dictionary, return the value.  Otherwise, reduce the level of be belief.  
        Continue this process until the key indexes the true parameter, not the believed parameter.
        """
        key = self.tolongform(key)

        if "?" in key:
            raise ValueError("You can set items via '?', not get items via '?'.")

        if key in self.contextual.external_refs:
            "If the value could be found in the ContextualDict"
            try: return {"mean": self.contextual[key], "var": self._defaults[self._parameter_in_key(key=key)]["var"]}
            except: return {"mean": self.contextual[key], "var": 0.2}

        "Check if the key is already in the dictionary"
        if key in self:
            value = super().__getitem__(key)
            if callable(value): value = value()
            return value

        while "believes:" in key:
            key = self.reduce_belief_level(key)
            if key in self:
                return self[key]

        keylst = self.key_to_list(key)
        parameter = self._parameter_in_key(key)
        if parameter == "values":
            if keylst[-3] == keylst[-1]:
                return self._defaults["iVi"]
            else: return self._defaults["iVj"]
        else: return self._defaults[parameter]

    def __setitem__(self, key: str, value: number_type | list[number_type, number_type] | dict[str: number_type]) -> None:
        """
        Sets the value of the item in a standard format even if the input value comes in varying data types.

        The value will end up looking like {'mean': 0.8, 'var': 0.2}, even 
        if it is entered as a number, iterable of numbers, or a dictionary.
        """
        key = self.tolongform(key)

        if not self.key_properly_formatted(key):
            raise ValueError(f"key improperly formatted: {key}")
        
        if isinstance(value, str) and "{" in value:
            "If the value is intended for a ContextualDict"
            self.contextual[key] = value
        else:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                "If you are too lazy to write a dictionary, (mean, var) will work."
                value = {"mean": value[0], "var": value[1]}

            elif isinstance(value, dict):
                "If you enter only partial information, fill in additional information with default values."
                if "var" in value and "mean" in value:
                    if value["mean"] is None:
                        value["mean"] = self._defaults[parameter]["mean"]
                    if value["var"] is None:
                        value["var"] = self._defaults[parameter]["var"]
                elif "var" in value and "mean" not in value:
                    value = {"mean": self._defaults[parameter]["var"], "var": value["var"]}
                elif "mean" in value and "var" not in value:
                    value = {"mean": value["mean"], "var": self._defaults[parameter]["var"]}

            keys = self.itterkey(key)       
            for key_ in keys:
                if isinstance(value, (int, float)):
                    parameter = self._parameter_in_key(key_)
                    if parameter == "values":
                        keylst = self.key_to_list(key)
                        if keylst[-3] == keylst[-1]:
                            parameter = "iVi"
                        else: parameter = "iVj"
                    value = {"mean": float(value), "var": self._defaults[parameter]["var"]}
                super().__setitem__(key_, value)

    def randomize(self, key: str = 'all', belief_key_prefix: str = "", distribution_type: str = 'normal', 
            interval: list[int] = [-3, 3], coefficients: list[int] = [1.0, 0.2]) -> None:
        """Creates parameter values by randomly sampling a distribution.
        
        Arguments:
            • key: string; Key to set a random value for.  If key == 'all', this randomizes all values.
                Can also randomize specific parameters: 'values', 'depth', 'risk', 'loss', 'rtime', 'dtime'.
            • belief_key_prefix: string; Prefix for key or keys to set a random value for, such as '', '1b', or'3b2b'.
            • distribution_type: Supports linear, sigmoid, uniform, triangle, normal, and beta distributions.
            • interval: list of two numbers; the low and high values on the x-axis to sample from
            • coefficients: list of two numbers; coefficients depends on the distribution type.
                Note: See Distributions.py for documentation about sampling from distributions.
        """
        import Distributions as dst
        param, all_keys = self._parameter_in_key(key), []
        
        if key == "all": 
            for param_type in self.param_types:
                self.randomize(key=param_type, belief_key_prefix=belief_key_prefix, distribution_type=\
                    distribution_type, interval=interval, coefficients=coefficients)
                
        if key == "values":
            for uid1 in self.uuids:
                for uid2 in self.uuids:
                    all_keys.append(f"{belief_key_prefix}{uid1}values{uid2}")

        elif param in self.param_types:
            for uid1 in self.uuids:  
                all_keys.append(f"{belief_key_prefix}{uid1}{param}")    

        else: all_keys.append(self.tolongform(belief_key_prefix+key))

        if distribution_type == "default":
            "If distribution type is 'default' then it samples a normal distribution around the mean, which is the default value."
            distribution_type, interval, coefficients = "normal", [-3, 3], [self._defaults[key], coefficients[1]]

        samples = dst.sample_distribution(distribution_type=distribution_type, 
            interval=interval, coefficients=coefficients, size=len(all_keys)).tolist()
        for key_, sample in zip(all_keys, samples): self[key_] = round(sample, 4)

    def setmatrix(self, matrix: list[list[float | int]] = [[]], 
                  belief_key_prefix: str = "", print_template: bool = False) -> None:
        """
        Allows user to simultaneously input multiple iVj values, instead of individually specifying each one.
        
        example_matrix = [
            [ 1.0, 0.8, 0.2],
            [ 0.8, 1.0, 0.1],
            [-0.4, 0.0, 1.0]]
        """
        
        if print_template:
            for idx in range(len(self.user_numbers)):
                print([0.0 for pdx in range (len(self.user_numbers))])
            exit()
        try:
            for idx, row in enumerate(matrix):
                for jdx, val in enumerate(row):
                    self[f"{belief_key_prefix}{idx}v{jdx}"] = val
                    
        except: raise ValueError("matrix must be a 2D list or 2D array of values.")   

    def totext(self, key: str = "all") -> None:
        """
        Converts items into English sentences to explain what they mean to developers
        """

        if key == "all": key = self.keys_
        if isinstance(key, str): key = [key]
        if not (isinstance(key, list) and all(isinstance(key_, str) for key_ in key)):
            raise ValueError(f"key type {type(key)} not supported.  Use string or list of strings.")
        
        for key_ in key:
            keylst = self.key_to_list(self.toshortform(key_))
            param, text = self._parameter_in_key(key_), "P"

            for idx, subkey in enumerate(keylst):
                if (0 < idx < len(keylst) - 2): text += "p" 
                if idx < len(keylst) - 3:
                    text += f"layer {subkey} believes that "
                elif idx == len(keylst) - 1:
                    if param == "values":
                        text += f"layer {keylst[-3]} values player {keylst[-1]} at {self[key_]['mean']}."
                    else: text += f"layer {keylst[-3]} {param_meanings[param]} {self[key_]['mean']}."

            print(text)

    def __repr__(self) -> str:
        """
        Sorting and rounding values when printing.
        """
        param_dict = {param: {} for param in self.param_types}
        for key in self.keys_:
            key = self.toshortform(key)
            param_dict[self._parameter_in_key(key)][key] = {"mean": round(self[key]["mean"], 2), "var": round(self[key]["var"], 2)}

        return pp.pformat({key: val for key, val in param_dict.items() if val}, sort_dicts=False)


class Players(ContextualDict):
    """
    Players is a smart dictionary that contains all information about players, both human and artificial.

    Based on the format of the key, it will retrieve information from one of its two inner dictionaries:
        • agent_parameters: PlayerParameters; Contains parameters of agents that determine their responses,
            like risk aversion and reasoning depth.  Keys must include the belief prefix and the parameter:
        • players_data: ParticipantsData; Contains all other information about each player, including uuid, 
            sid id, avatar appearance, etc. 

    The purpose of Players is to store everything you need to know about players in one place, which
    will be in the Experiment instance, which will also be stored at the root of each game tree.        
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_parameters = PlayerParameters(players=ParticipantsData())
        self.experimenter = None

    def __setitem__(self, key, value):
        """
        Sets the item in the correct inner dictionary based on features of the key.
        """
        if self.isuuid(key=key) or self.issid(key=key):
            self.players_data[key] = value
        elif self.hasparam(key=key):
            self.agent_parameters[key] = value
        else: super().__setitem__(key, value)

    def __getitem__(self, key):
        """
        Gets the item in the correct inner dictionary based on features of the key.
        """
        if isinstance(key, slice):
            return [self[k] for k in self.uuids[key]]
        elif self.isuuid(key=key) or self.issid(key=key):
            return self.players_data[key]
        elif self.hasparam(key=key):
            return self.agent_parameters[key]
        else: return super().__getitem__(key)

    def __getattr__(self, name):
        if name == "nplayers":
            return len(self.agent_parameters.players_dict.user_ids)
        if name == "n_humans":
            return len([player for player in self.agent_parameters.players_dict.values() if player['player_type'] != 'robot'])
        if name == "n_robots":
            return len([player for player in self.agent_parameters.players_dict.values() if player['player_type'] == 'robot'])
        if name == "user_numbers":
            return list(range(len(self.agent_parameters.players_dict.user_ids)))
        if name == "uuids" or name == "user_ids":
            return self.agent_parameters.players_dict.user_ids
        if name == "players_data" or name == "players_dict":
            return self.agent_parameters.players_dict
        if name == "players_list":
            return list(self.agent_parameters.players_dict.values())

    def isuuid(self, key: str | uuid.UUID | int) -> bool:
        """
        Determines of the key is a uuid.
        """
        if isinstance(key, int): 
            return True
        if isinstance(key, uuid.UUID): return True
        if "-" in key:
            if [len(subuuid) for subuuid in key.split("-")] == [8, 4, 4, 4, 12]:
                return True
        return False

    def issid(self, key: str | int) -> bool:
        """
        Determines of the key is a sid id.
        """
        if isinstance(key, int): 
            return True
        if isinstance(key, str):
            if "http" in key or "wws://" in key or "sid" in key:
                return True
        return False

    def hasparam(self, key: str) -> bool:
        """
        Determines if a parameter is contained in the key
        """
        if isinstance(key, str): 
            for substr in self.agent_parameters.param_types + ["?"]:
                if substr in key: return True        
            if "v" in key and key[-1].isdigit():
                vsplit = key.split("v")
                if len(vsplit) > 1 and vsplit[-1].isdigit() \
                    and vsplit[-2][-1].isdigit(): return True   
        else: return False

    def add_players(self, number: int = None, username: str = None, email_address: str = None, 
            uuid_: str = None, sid_: str = None, player_type: str = 'robot', avatar_shape: str = None, avatar_color: str = None) -> None:
        """
        Add players to the dictionary.  If number is an integer, this generates that number of random
        agents.  Otherwise, this method will add specific a specific player based on the other arguments.
        """
        if isinstance(number, int):
            for idx in range(number):
                self.agent_parameters.players_dict.create_player()
        else: self.agent_parameters.players_dict.create_player(username=username, email_address=email_address, 
                                                               uuid_=uuid_, sid_=sid_, player_type=player_type, 
                                                               avatar_shape=avatar_shape, avatar_color=avatar_color)

    def for_experiment(self, participants: set[uuid.UUID], experimenter_uuid: str = None, 
                       experimenter_sid: str = None, start_date: str = None) -> None:
        """
        Restricts the dictionary only to current participants.  Does not work yet.
        """
        if start_date is None: start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        if experimenter_sid is None: experimenter_sid = str(uuid.uuid4())
        if experimenter_uuid is None: experimenter_uuid = str(uuid.uuid4())

        self.experimenter = self["experimenter"] = {"uuid": experimenter_uuid, "sid": experimenter_sid}
        self.expid = self["expid"] = f"EXPID-{experimenter_uuid}-{start_date}"

        self.players_data.to_experiment(uuids=participants, 
            experimenter_uuid=experimenter_uuid, start_date=start_date)

    def params_by_tree(self, tree: dict, key: str):
        """
        Lookup parameters but using player numbers from the game tree, not the list of uuids in the Players dict.
        
        Example:
            If these are the uuids in the Players dict: ['uuid0', 'uuid1', 'uuid2', 'uuid3', 'uuid4']
            and if these are the uuids in the tree ['uuid1', 'uuid3', 'uuid0'], then the key '0v2' returns
            How much player 'uuid1' values player 'uuid0', not how much player 'uuid0' values player 'uuid2'.
        """
        def istree(tree) -> bool:
            "Same as isinstance(tree, Tree) without importing GameTree.py"
            for attr in ["nplayers", "players", "idnum", "options"]:
                if not hasattr(tree, attr): return False
            return True

        if not istree(tree): raise ValueError("tree must be an instance of Tree.")

        key = self.agent_parameters.toshortform(key)
        keylst = self.agent_parameters.key_to_list(key)
        user_ids = [plr.uuid for plr in tree.players]

        beginning = "".join([f"{user_ids[int(ele)]}believes:" if len(ele) < 32 and ele.isdigit() else f"{ele}believes:" for ele in keylst[:-3]])
        ending = "".join([user_ids[int(ele)] if len(ele) < 32 and ele.isdigit() else ele for ele in keylst[-3:]])

        return self[beginning + ending]

    def parameters_for(self, player: str | int, short_form: bool = True) -> dict:
        """
        Returns the parameters for player in a dictionary.
        
        Arguments:
            • player: str | int; The player number of uuid of the player of interest.
            • short_form: bool; If True, the keys in the returned dictionary will be abbreviated.

        Returns:
            • dict: Dictionary of the parameters fo a specific player
        """
        if isinstance(player, int) and not short_form:
            try: player = self.uuids[player]
            except IndexError:
                raise ValueError(f"player {player} has not yet been added to this dictionary.")
        elif isinstance(player, str) and short_form:
            try: player = self.uuids.index(player)
            except ValueError:
                raise ValueError(f"player {player} has not yet been added to this dictionary.")            
        paramsfor = {}
        for param in self.agent_parameters.param_types:
            if param != "values":
                pkey = f"{player}{param}"
                paramsfor[pkey] = self[pkey]
            else:
                for idx, plr in enumerate(self.uuids):
                    if short_form:
                        if idx != player:
                            vkey = f"{player}v{idx}"
                            keyv = f"{idx}v{player}"
                            paramsfor[vkey] = self[vkey]
                            paramsfor[keyv] = self[keyv]
                    else:
                        if plr != player:
                            vkey = f"{player}values{idx}"
                            keyv = f"{idx}values{player}"
                            paramsfor[vkey] = self[vkey]
                            paramsfor[keyv] = self[keyv]
        return paramsfor

    def make_rational(self, player: str | int | uuid.UUID, iVj: float = 0.0) -> None:
        """
        Sets the parameters of a player to make that player rational, meaning that this player has no risk 
        aversion, loss aversion, envy, or guilt and has a reasoning depth parameter of 100.  By default, this 
        player will be rationally self-interested, but setting iVj to 1.0 can make them utilitarian.  Such a 
        rational player is useful as a reference point because their behavior is easier to interpret and for 
        the same reason such a player is useful for debugging.
        
        Arguments:
            • player: int | str | uuid; The player number of uuid of the player of interest.
            • iVj: float; The parameter for how much this player cares about others.
        """
        if isinstance(player, int):
            if player > len(self.uuids):
                raise ValueError(f"player {player} has not yet been added to this dictionary.")
        elif isinstance(player, (str, uuid.UUID)):
            try: player = self.uuids.index(player)
            except ValueError:
                raise ValueError(f"player {player} has not yet been added to this dictionary.")    
        else: TypeError(f"player type {type(player)} not supported. Use int, str, or uuid.")

        rational_parameters = {
            "v?": iVj, 
            f"v{player}": 1.0,  
            "rdepth": 100,
            "risk": 1.0, 
            "loss": 1.0,
            "yvne": 0.0, 
            "envy": 0.0
            }

        for param in list(rational_parameters.keys()):
            self[f"{player}{param}"] = rational_parameters[param]