# lite server 版本信息汇总

- 服务器架构：aarch64

```cmd
uname -m
```

- 操作系统：EulerOS

```cmd
uname -a
``` 

- npu：910B4（或910B）
- npu驱动：24.1.0.3

```cmd
npu-smi info
```

- python：3.9.9

```cmd
python --version
```

- Ascend-toolkit：24.1.0.3

```cmd
cat /usr/local/Ascend/version.info
```

- pytorch：2.1.0

```cmd
python3 -c "import torch; print(torch.__version__)"
```

- cmake：3.22.0

```cmd
 cmake --version
```
