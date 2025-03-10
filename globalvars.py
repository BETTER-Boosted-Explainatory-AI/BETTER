imagenet_labels = [
    'African_hunting_dog', 'Arctic_fox', 'French_bulldog', 'Gordon_setter', 'Ibizan_hound', 'Newfoundland', 'Saluki', 'Tibetan_mastiff', 
    'Walker_hound', 'aircraft_carrier', 'ant', 'ashcan', 'barrel', 'beer_bottle', 'black-footed_ferret', 'bolete', 'bookshop', 'boxer',
    'cannon', 'carousel', 'carton', 'catamaran', 'chime', 'cliff', 'clog', 'cocktail_shaker', 'combination_lock', 'consomme', 
    'coral_reef', 'crate', 'cuirass', 'dalmatian', 'dishrag', 'dome', 'dugong', 'ear', 'electric_guitar', 'file', 'fire_screen', 
    'frying_pan', 'garbage_truck', 'golden_retriever', 'goose', 'green_mamba', 'hair_slide', 'harvestman', 'holster',
    'horizontal_bar', 'hotdog', 'hourglass', 'house_finch', 'iPod', 'jellyfish', 'king_crab', 'komondor', 'ladybug', 'lion',
    'lipstick', 'malamute', 'meerkat', 'miniature_poodle', 'miniskirt', 'missile', 'mixing_bowl', 'nematode', 'oboe', 
    'orange', 'organ', 'parallel_bars', 'pencil_box', 'photocopier', 'poncho', 'prayer_rug', 'reel', 'rhinoceros_beetle', 
    'robin', 'rock_beauty', 'school_bus', 'scoreboard', 'slot', 'snorkel', 'solar_dish', 'spider_web', 'stage', 
    'street_sign', 'tank', 'theater_curtain', 'three-toed_sloth', 'tile_roof', 'tobacco_shop', 'toucan', 'triceratops', 
    'trifle', 'unicycle', 'upright', 'vase', 'white_wolf', 'wok', 'worm_fence', 'yawl'
    ]

imagenet_labels_dict = {'n02074367': 'dugong', 'n03062245': 'cocktail_shaker', 'n06794110': 'street_sign', 'n03854065': 'organ', 'n03888605': 'parallel_bars', 'n01930112': 'nematode', 'n02120079': 'Arctic_fox', 'n03127925': 'crate', 'n01532829': 'house_finch', 'n03337140': 'file', 'n04243546': 'slot', 'n02114548': 'white_wolf', 'n03908618': 'pencil_box', 'n02101006': 'Gordon_setter', 'n03075370': 'combination_lock', 'n02108089': 'boxer', 'n02966193': 'carousel', 'n07613480': 'trifle', 'n04509417': 'unicycle', 'n01558993': 'robin', 'n03838899': 'oboe', 'n02950826': 'cannon', 'n01770081': 'harvestman', 'n04258138': 'solar_dish', 'n09246464': 'cliff', 'n02871525': 'bookshop', 'n02747177': 'ashcan', 'n02110341': 'dalmatian', 'n02823428': 'beer_bottle', 'n03207743': 'dishrag', 'n02443484': 'black-footed_ferret', 'n03400231': 'frying_pan', 'n02110063': 'malamute', 'n02687172': 'aircraft_carrier', 'n02795169': 'barrel', 'n02099601': 'golden_retriever', 'n07584110': 'consomme', 'n03220513': 'dome', 'n07697537': 'hotdog', 'n04251144': 'snorkel', 'n03476684': 'hair_slide', 'n03527444': 'holster', 'n02981792': 'catamaran', 'n03980874': 'poncho', 'n07747607': 'orange', 'n04149813': 'scoreboard', 'n03146219': 'cuirass', 'n02091831': 'Saluki', 'n04389033': 'tank', 'n03770439': 'miniskirt', 'n02108551': 'Tibetan_mastiff', 'n02219486': 'ant', 'n03017168': 'chime', 'n03773504': 'missile', 'n01910747': 'jellyfish', 'n09256479': 'coral_reef', 'n04275548': 'spider_web', 'n04612504': 'yawl', 'n04515003': 'upright', 'n03417042': 'garbage_truck', 'n04296562': 'stage', 'n03924679': 'photocopier', 'n02111277': 'Newfoundland', 'n01704323': 'triceratops', 'n01981276': 'king_crab', 'n04596742': 'wok', 'n04146614': 'school_bus', 'n02116738': 'African_hunting_dog', 'n04522168': 'vase', 'n01843383': 'toucan', 'n02165456': 'ladybug', 'n03047690': 'clog', 'n04443257': 'tobacco_shop', 'n04435653': 'tile_roof', 'n03535780': 'horizontal_bar', 'n02108915': 'French_bulldog', 'n03775546': 'mixing_bowl', 'n02129165': 'lion', 'n02174001': 'rhinoceros_beetle', 'n04067472': 'reel', 'n03347037': 'fire_screen', 'n13054560': 'bolete', 'n02971356': 'carton', 'n03272010': 'electric_guitar', 'n03544143': 'hourglass', 'n01749939': 'green_mamba', 'n02606052': 'rock_beauty', 'n04418357': 'theater_curtain', 'n03584254': 'iPod', 'n02105505': 'komondor', 'n02457408': 'three-toed_sloth', 'n01855672': 'goose', 'n04604644': 'worm_fence', 'n13133613': 'ear', 'n02138441': 'meerkat', 'n02091244': 'Ibizan_hound', 'n03998194': 'prayer_rug', 'n02089867': 'Walker_hound', 'n03676483': 'lipstick', 'n02113712': 'miniature_poodle'}

cifar100_labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]