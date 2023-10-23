# Code inspired from Confluent Cloud official examples library
# https://github.com/confluentinc/examples/blob/7.1.1-post/clients/cloud/python/producer.py

from confluent_kafka import Producer
import json
import ccloud_lib # Library not installed with pip but imported from ccloud_lib.py
import numpy as np
import time
import requests

# Initialize configurations from "python.config" file
CONF = ccloud_lib.read_ccloud_config("python.config")
TOPIC = "prediction_payments" 

# Create Producer instance
producer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF)
producer = Producer(producer_conf)

# Create topic if it doesn't already exist
ccloud_lib.create_topic(CONF, TOPIC)

try:
    while True:
        record_key = "prediction"
        record_value = requests.get("https://localhost:5001/prediction")
        print("Producing record: {}\t{}".format(record_key, record_value.content))
        if record_value.status_code == 200:
            data = record_value.json()  # Convertit la réponse JSON en Python
        # Traitez les données comme vous le souhaitez
        else:
            print(f"Erreur lors de la requête : {record_value.status_code}")

        # This will actually send data to your topic
        producer.produce(
            TOPIC,
            key=record_key,
            value=record_value.content,
        )
        time.sleep(12)
# Interrupt infinite loop when hitting CTRL+C
except KeyboardInterrupt:
    pass
finally:
    producer.flush() # Finish producing the latest event before stopping the whole script