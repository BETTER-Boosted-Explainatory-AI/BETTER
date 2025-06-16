user = {
    "user_id": "8482bdbf-fada-4a24-a45d-15ff1498b519",
    "email": "test@example.com",
    "password": "TestP@ssw0rd"
}

mock_dendrogram = {
    "id": 1003,
    "name": "whole_121",
    "children": [
        {
            "id": 1001,
            "name": "whole_116",
            "children": [
                {
                    "id": 283,
                    "name": "Persian_cat"
                },
                {
                    "id": 1000,
                    "name": "bathtub_2",
                    "children": [
                        {
                            "id": 435,
                            "name": "bathtub"
                        },
                        {
                            "id": 876,
                            "name": "tub"
                        }
                    ],
                    "value": 0.0
                }
            ],
            "value": 0.7073672218248248
        },
        {
            "id": 1002,
            "name": "domestic_cat",
            "children": [
                {
                    "id": 281,
                    "name": "tabby"
                },
                {
                    "id": 285,
                    "name": "Egyptian_cat"
                }
            ],
            "value": 0.7376661213347688
        }
    ],
    "value": 1.2287538811527459
}

mock_sub_dendrogram = {
    "id": 1000,
    "name": "bathtub_2",
    "children": [
        {
            "id": 435,
            "name": "bathtub"
        },
        {
            "id": 876,
            "name": "tub"
        }
    ],
    "value": 0.0
}

selected_labels = ["bathtub", "tub"]