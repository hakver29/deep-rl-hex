import hexclient.BasicClientActor as BSA

client = BSA.BasicClientActor(load_best_model=True)
client.connect_to_server()