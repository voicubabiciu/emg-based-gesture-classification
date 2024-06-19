from influxdb_client.client import influxdb_client


class DatabaseService:
    def __init__(self, bucket, org, url, token, use_database):
        self.use_database = use_database
        self.bucket = bucket
        self.org = org
        self.url = url
        self.token = token
        if self.use_database == 'True':
            self.dbClient = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
        else:
            self.dbClient = None

    def store_dataset(self, data):
        if self.dbClient:
            self.dbClient.write_api()._write_batching(bucket=self.bucket, org=self.org, data=data)
