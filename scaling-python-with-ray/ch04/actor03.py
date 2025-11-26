import ray
import os
from ray.util import ActorPool

ray.init(num_cpus=4)

class FilePersistence:
    def __init__(self, basedir: str = '.'):
        self.basedir = basedir
        os.makedirs(self.basedir, exist_ok=True)

    def exists(self, key:str) -> bool:
        return os.path.exists(self.basedir + '/' + key)

    def save(self, key: str, data: dict):
        bytes_data = ray.cloudpickle.dumps(data)
        with open(self.basedir + '/' + key, "wb") as f:
            f.write(bytes_data)

    def restore(self, key:str) -> dict:
        if not self.exists(key):
            return None
        else:
            with open(self.basedir + '/' + key, "rb") as f:
                bytes_data = f.read()
            return ray.cloudpickle.loads(bytes_data)

@ray.remote
class Account:
    def __init__(self, account_key: str):
        self.persistence = FilePersistence(basedir="./data")
        self.key = account_key
        self.balance = 0.0
        self.minimal = 0.0
        self.restorestate()

    def initialize(self, balance: float, minimal_balance: float):
        if balance < minimal_balance:
            raise Exception("Starting balance is less than minimal balance")
        self.balance = balance
        self.minimal = minimal_balance
        self.storestate()

    def get_balance(self) -> float:
        return self.balance

    def deposit(self, amount: float) -> float:
        if amount < 0:
            raise Exception("Cannot deposit negative amount")
        self.balance = self.balance + amount
        self.storestate()
        return self.balance

    def withdraw(self, amount: float) -> float:
        if amount < 0:
            raise Exception("Cannot withdraw negative amount")
        balance = self.balance - amount
        if balance < self.minimal:
            raise Exception("Withdrawal is not supported by current balance")
        self.balance = balance
        self.storestate()
        return balance

    def restorestate(self) -> bool:
        state = self.persistence.restore(self.key)
        if state is not None:
            self.balance = state['balance']
            self.minimal = state['minimal']
            return True
        else:
            return False

    def storestate(self):
        self.persistence.save(self.key,
                    {'balance': self.balance, 'minimal': self.minimal})

# 创建 ActorPool 管理多个账户
accounts = [
    Account.remote(account_key=f"account_{i}") 
    for i in range(3)
]

# 初始化各个账户
for i, account in enumerate(accounts):
    ray.get(account.initialize.remote(balance=100.0 + i*50, minimal_balance=20.0))

# 初始化 ActorPool
pool = ActorPool(accounts)

# 使用 ActorPool 并行处理 - 修正：不需要再调用 ray.get
print("=== 使用 ActorPool 获取所有账户余额 ===")
results = list(pool.map(lambda actor, value: actor.get_balance.remote(), [None]*3))
print("All account balances:", results)

# 演示并行操作
print("\n=== 并行存款操作 ===")
deposit_results = list(pool.map(lambda actor, value: actor.deposit.remote(value), [10, 20, 30]))
print("After deposits:", deposit_results)

print("\n=== 并行取款操作 ===")
withdraw_results = list(pool.map(lambda actor, value: actor.withdraw.remote(value), [15, 25, 35]))
print("After withdrawals:", withdraw_results)

# 也可以单独操作某个账户
print(f"\n=== 单独操作账户 0 ===")
print(f"Account 0 balance: {ray.get(accounts[0].get_balance.remote())}")

# 最终所有账户余额
print("\n=== 最终所有账户余额 ===")
final_balances = list(pool.map(lambda actor, value: actor.get_balance.remote(), [None]*3))
print("Final balances:", final_balances)
