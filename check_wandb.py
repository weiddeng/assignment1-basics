import wandb

try:
    print("Attempting handshake...")
    wandb.init(project="test-connection", mode="online")
    print("✅ SUCCESS! Logged in as:", wandb.run.entity)
    wandb.finish()
except Exception as e:
    print("❌ FAILED. You are not logged in.")
    print(e)