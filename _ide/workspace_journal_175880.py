# 2025-11-28T09:55:30.602966445
import vitis

client = vitis.create_client()
client.set_workspace(path="forward")

vitis.dispose()

