from tasks import app
import sys
import os

if __name__ == '__main__':
    master_ip = None
    worker_name = None

    if len(sys.argv) >= 3:
        master_ip = sys.argv[1]
        worker_name = sys.argv[2]
    else:
        master_ip = os.environ.get('REDIS_HOST') or os.environ.get('BROKER_HOST')
        worker_name = os.environ.get('WORKER_NAME')

    if not master_ip or not worker_name:
        print("Error: Please provide the MASTER_IP and WORKER_NAME as command-line arguments or set REDIS_HOST and WORKER_NAME.")
        sys.exit(1)

    redis_port = os.environ.get('REDIS_PORT', '6379')
    app.conf.broker_url = f'redis://{master_ip}:{redis_port}/0'
    print(f'redis://{master_ip}:{redis_port}/0')

    app.worker_main(['worker', '-n', worker_name, '--loglevel=info'])
    