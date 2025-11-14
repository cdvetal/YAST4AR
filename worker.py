from tasks import app
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Error: Please provide the MASTER_IP and WORKER_NAME as command-line arguments.")
        sys.exit(1)

    app.conf.broker_url = f'redis://{sys.argv[1]}:6379/0'
    print(f'redis://{sys.argv[1]}:6379/0')

    worker_name = sys.argv[2]

    app.worker_main(['worker', '-n', worker_name, '--loglevel=info'])
    