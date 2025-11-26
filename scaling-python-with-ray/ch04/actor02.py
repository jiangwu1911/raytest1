import ray
import os

ray.init(num_cpus=4)

class BasePersitence:
    def exists(self, key:str) -> bool:
        pass
    def save(self, key: str, data: dict):
        pass
    def restore(self, key:str) -> dict:
        pass

class FilePersistence(BasePersitence):
    def __init__(self, basedir: str = '.'):
        self.basedir = basedir
        # 确保目录存在
        os.makedirs(self.basedir, exist_ok=True)

    def exists(self, key:str) -> bool:
        return os.path.exists(self.basedir + '/' + key)  # 修正：使用 os.path.exists

    def save(self, key: str, data: dict):
        bytes = ray.cloudpickle.dumps(data)
        with open(self.basedir + '/' + key, "wb") as f:
            f.write(bytes)

    def restore(self, key:str) -> dict:
        if not self.exists(key):
            return None
        else:
            with open(self.basedir + '/' + key, "rb") as f:
                bytes = f.read()
            return ray.cloudpickle.loads(bytes)

@ray.remote
class Account:
    def __init__(self, balance: float, minimal_balance: float, account_key: str,
                 persistence: BasePersitence):
        self.persistence = persistence
        self.key = account_key
        if not self.restorestate():
            if balance < minimal_balance:
                raise Exception("Starting balance is less than minimal balance")
            self.balance = balance
            self.minimal = minimal_balance
            self.storestate()

    def get_balance(self) -> float:  # 修正：避免命名冲突
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
        if state != None:
            self.balance = state['balance']
            self.minimal = state['minimal']
            return True
        else:
            return False

    def storestate(self):
        self.persistence.save(self.key,
                    {'balance' : self.balance, 'minimal' : self.minimal})

# 创建持久化实例
persistence = FilePersistence(basedir="./data")

# 创建actor时传递persistence参数
account_actor = Account.remote(
    balance=100., 
    minimal_balance=20., 
    account_key="jwu",
    persistence=persistence
)

print(f"Current balance {ray.get(account_actor.get_balance.remote())}")
print(f"New balance {ray.get(account_actor.withdraw.remote(40.))}")
print(f"New balance {ray.get(account_actor.deposit.remote(30.))}")
