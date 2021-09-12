# encoding=gbk

import pika
import time
import json


class Producer(object):
    def __init__(self, ip, user, pwd):
        auth = pika.PlainCredentials(user, pwd)
        con_params = pika.ConnectionParameters(host=ip,
                                  port=5672,
                                  virtual_host='/',
                                  credentials=auth)
        self.connection = pika.BlockingConnection(con_params)
        self.channel = self.connection.channel()

    def close_connection(self):
        self.connection.close()

    def send_msg(self, quene_name, message):
        self.channel.queue_declare(queue=quene_name,durable=True)  # declare queue
        # 交换机  topic.mode.capture.image
        self.channel.basic_publish(exchange='', routing_key=quene_name, body=message)  # the body is the msg content
        # self.close_connection()

    def send_filter(self):
        pass

    def send_beauty(self):
        pass


class Consumer(object):
    def __init__(self, ip, user, pwd):
        auth = pika.PlainCredentials(user, pwd)
        con_params = pika.ConnectionParameters(host=ip,
                                               port=5672,
                                               virtual_host='/',
                                               credentials=auth)
        self.connection = pika.BlockingConnection(con_params)
        self.channel = self.connection.channel()

    def eat_msg(self, quene_name):
        self.channel.queue_declare(queue=quene_name,durable=True)  # decalre queue

        def callback(ch, method, properties, body):
            message = json.loads(body)

            print(message)
            print('---------完成')

        self.channel.basic_consume(quene_name, callback, False)
        self.channel.start_consuming()


if __name__ == '__main__':
    # ip = '127.0.0.1'
    # ip = '192.168.31.239'
    #
    # mq_p = Producer(ip, 'ng', 'ng')
    # msg = {
    #     "timestamp": time.time(),
    #     "condition": '10001'
    # }
    # # msg = 'hello'
    # mq_p.send_msg("image",  json.dumps(msg))
    ip = '127.0.0.1'

    mq_c = Consumer(ip, 'guest', 'guest')
    mq_c.eat_msg('flow.judge')
