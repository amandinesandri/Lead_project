# Example written based on the official 
# Confluent Kakfa Get started guide https://github.com/confluentinc/examples/blob/7.1.1-post/clients/cloud/python/consumer.py

from confluent_kafka import Consumer
import json
import ccloud_lib
import time
import requests

def consume_kafka_message():
    # Initialize configurations from "python.config" file
    CONF = ccloud_lib.read_ccloud_config("python.config")
    TOPIC = "prediction"

    # Create Consumer instance
    consumer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF)
    consumer_conf['group.id'] = 'payments_consumer'
    consumer_conf['auto.offset.reset'] = 'earliest'
    consumer = Consumer(consumer_conf)

    # Subscribe to topic
    consumer.subscribe([TOPIC])

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                print("Waiting for message or event/error in poll()")
                continue
            elif msg.error():
                if msg.error().code() == KafkaException._PARTITION_EOF:
                    # End of partition event, not an error
                    print("Got End of Partition event")
                    continue
                else:
                    print(f'Error: {msg.error()}')
            else:
                record_value = msg.value()
                data = json.loads(record_value)
                prediction = data
                print(f">>>>>> {prediction} <<<<<<")
                return prediction
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

# Call the function to consume a Kafka message
message = consume_kafka_message()
print("Returned message:", message)