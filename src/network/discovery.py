"""
局域网设备发现模块
"""

import socket
import os
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """设备信息"""
    name: str
    ip: str
    port: int
    status: str = "idle"


class DeviceDiscovery:
    """局域网设备发现"""

    def __init__(
        self,
        port: int,
        service_type: str = "_airgesture._tcp.local.",
        single_machine_mode: bool = False
    ):
        """
        初始化设备发现

        Args:
            port: 服务端口
            service_type: 服务类型
            single_machine_mode: 单机调试模式
        """
        self.port = port
        self.service_type = service_type
        self.single_machine_mode = single_machine_mode
        self.devices: Dict[str, DeviceInfo] = {}
        self._zeroconf = None
        self._service_info = None
        self._browser = None

    def start(self):
        """启动服务注册和发现"""
        if self.single_machine_mode:
            # 单机模式：不使用 zeroconf
            print("单机调试模式：跳过设备发现")
            # 单机模式下添加本机设备
            self.add_device("LocalHost", "127.0.0.1", self.port, "ready")
            return

        try:
            from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser
            import threading

            # 获取本机 IP
            local_ip = self._get_local_ip()
            hostname = os.environ.get('COMPUTERNAME', 'Unknown')

            # 注册服务（在后台线程中执行，避免阻塞）
            def register_service():
                try:
                    self._zeroconf = Zeroconf()
                    self._service_info = ServiceInfo(
                        self.service_type,
                        f"AirGesture-{hostname}.{self.service_type}",
                        addresses=[socket.inet_aton(local_ip)],
                        port=self.port,
                        properties={"name": hostname, "status": "idle"}
                    )
                    self._zeroconf.register_service(self._service_info)

                    # 启动服务发现
                    listener = _ServiceListener(self)
                    self._browser = ServiceBrowser(
                        self._zeroconf,
                        self.service_type,
                        listener
                    )
                    print(f"服务已注册: AirGesture-{hostname} @ {local_ip}:{self.port}")
                except Exception as e:
                    print(f"服务注册失败: {e}")

            # 在后台线程中启动服务
            reg_thread = threading.Thread(target=register_service, daemon=True)
            reg_thread.start()

            # 添加本机设备（以便单机测试）
            self.add_device(hostname, local_ip, self.port, "ready")

        except ImportError:
            print("警告: 未安装 zeroconf，设备发现功能不可用")
            print("请运行: pip install zeroconf")
            # 添加本机设备作为回退
            local_ip = self._get_local_ip()
            hostname = os.environ.get('COMPUTERNAME', 'Unknown')
            self.add_device(hostname, local_ip, self.port, "ready")
        except Exception as e:
            print(f"设备发现启动失败: {e}")
            # 添加本机设备作为回退
            local_ip = self._get_local_ip()
            hostname = os.environ.get('COMPUTERNAME', 'Unknown')
            self.add_device(hostname, local_ip, self.port, "ready")

    def stop(self):
        """停止服务"""
        if self._zeroconf:
            try:
                if self._service_info:
                    self._zeroconf.unregister_service(self._service_info)
                self._zeroconf.close()
            except:
                pass
            self._zeroconf = None

    def get_devices(self) -> List[DeviceInfo]:
        """
        获取可用设备列表

        Returns:
            设备列表
        """
        if self.single_machine_mode:
            # 单机模式：返回 localhost
            return [DeviceInfo(
                name="LocalHost",
                ip="127.0.0.1",
                port=self.port,
                status="ready"
            )]

        return list(self.devices.values())

    def add_device(self, name: str, ip: str, port: int, status: str = "idle"):
        """添加发现的设备"""
        key = f"{ip}:{port}"
        self.devices[key] = DeviceInfo(
            name=name,
            ip=ip,
            port=port,
            status=status
        )
        print(f"发现设备: {name} @ {ip}:{port}")

    def remove_device(self, name: str):
        """移除设备"""
        keys_to_remove = [k for k, v in self.devices.items() if v.name == name]
        for key in keys_to_remove:
            del self.devices[key]
            print(f"设备离线: {name}")

    def _get_local_ip(self) -> str:
        """获取本机 IP 地址"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"


class _ServiceListener:
    """zeroconf 服务监听器"""

    def __init__(self, discovery: DeviceDiscovery):
        self.discovery = discovery

    def add_service(self, zeroconf, service_type, name):
        """发现新服务"""
        try:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                ip = socket.inet_ntoa(info.addresses[0])
                port = info.port
                device_name = info.properties.get(b'name', b'Unknown').decode()
                status = info.properties.get(b'status', b'idle').decode()
                self.discovery.add_device(device_name, ip, port, status)
        except Exception as e:
            print(f"解析服务失败: {e}")

    def remove_service(self, zeroconf, service_type, name):
        """服务离线"""
        # 从名称中提取设备名
        device_name = name.replace(f".{service_type}", "").replace("AirGesture-", "")
        self.discovery.remove_device(device_name)

    def update_service(self, zeroconf, service_type, name):
        """服务更新"""
        self.add_service(zeroconf, service_type, name)
