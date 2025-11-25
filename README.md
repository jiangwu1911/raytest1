这些程序用来测试ray远程调用的功能，https://github.com/ray-project/ray/

client, server两边python版本，ray版本要完全一致

服务端

    $ pip install torch numpy
    
    $ pip install "ray[client,default,rllib,serve]"

    启动ray server
    $ ray start --head \
        --node-ip-address=192.168.1.217 \
        --port=6379 \
        --num-gpus=1 \
        --include-dashboard=true \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265

客户端

    $ pip install torch numpy
    
    $ pip install "ray[client,default,rllib,serve]"
