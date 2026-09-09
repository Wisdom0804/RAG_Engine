# -*- coding: utf-8 -*-
"""
文件名：snow_flake.py.py
文件描述: 
作者: 郑智文
创建日期: 2026/9/9 16:50
版本: 1.0
IDE: PyCharm

Copyright (c) 2026 星际区块链（深圳）有限公司. All rights reserved.
"""
import threading
import time

from app.config.security import secure


class SnowFlake:
    """
    雪花 ID 生成器（单例模式）

    64 位 ID 结构（从高位到低位）：
        | 1 bit 符号位 | 41 bits 毫秒时间戳 | 5 bits 数据中心ID | 5 bits 机器ID | 12 bits 序列号 |

        - 符号位：恒为 0，保证 ID 为正数
        - 时间戳：相对起始纪元（默认 2024-01-01）的毫秒数，可用约 69 年
        - 数据中心ID + 机器ID：共 10 位，最多支持 1024 个节点
        - 序列号：同一毫秒内的自增序号，单节点每毫秒最多生成 4096 个 ID

    线程安全：next_id 加锁，可在多线程环境安全调用。
    时钟回拨：检测到系统时钟倒退时，在容忍阈值内自旋等待时钟追平，
             超出阈值则抛出 RuntimeError，避免产生重复 ID。
    单例：模块级实例 snow_flake 即单例，直接 from snow_flake import snow_flake 使用。
    """

    # ---------- 位长度与位移配置 ----------
    # 序列号占用位数
    SEQUENCE_BITS = 12
    # 机器ID占用位数
    WORKER_ID_BITS = 5
    # 数据中心ID占用位数
    DATA_CENTER_ID_BITS = 5

    # 序列号掩码：12 位全 1，用于序列号回绕
    SEQUENCE_MASK = ~(-1 << SEQUENCE_BITS)  # 0xFFF = 4095

    # 机器ID最大值：31
    MAX_WORKER_ID = ~(-1 << WORKER_ID_BITS)
    # 数据中心ID最大值：31
    MAX_DATA_CENTER_ID = ~(-1 << DATA_CENTER_ID_BITS)

    # 机器ID左移位数：序列号长度
    WORKER_ID_SHIFT = SEQUENCE_BITS  # 12
    # 数据中心ID左移位数：序列号长度 + 机器ID长度
    DATA_CENTER_ID_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS  # 17
    # 时间戳左移位数：序列号长度 + 机器ID长度 + 数据中心ID长度
    TIMESTAMP_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS + DATA_CENTER_ID_BITS  # 22

    # 起始纪元：2026-01-01 00:00:00 UTC 的毫秒时间戳
    START_EPOCH = 1767225600000

    # 时钟回拨容忍阈值（毫秒），超过则判定为异常回拨
    CLOCK_BACKWARD_TOLERANCE = 5

    def __init__(self, worker_id: int = secure.MACHINE_ID, data_center_id: int = 0):
        """
        初始化雪花 ID 生成器

        单例通过模块级实例 snow_flake 提供，直接构造仍可用作多实例场景。

        :param worker_id: 机器ID，范围 [0, 31]，默认取 secure.MACHINE_ID
        :param data_center_id: 数据中心ID，范围 [0, 31]，默认 0
        """
        if worker_id < 0 or worker_id > self.MAX_WORKER_ID:
            raise ValueError(
                f"worker_id 越界，取值范围 [0, {self.MAX_WORKER_ID}]，当前: {worker_id}")
        if data_center_id < 0 or data_center_id > self.MAX_DATA_CENTER_ID:
            raise ValueError(
                f"data_center_id 越界，取值范围 [0, {self.MAX_DATA_CENTER_ID}]，当前: {data_center_id}")

        self.worker_id = worker_id
        self.data_center_id = data_center_id
        # 上次生成 ID 的时间戳（毫秒），初始为 -1 表示尚未生成过
        self._last_timestamp = -1
        # 当前毫秒内的序列号
        self._sequence = 0
        # next_id 的互斥锁，保证多线程下序列号分配的原子性
        self._lock = threading.Lock()

    def _current_millis(self) -> int:
        """返回当前毫秒时间戳"""
        return int(time.time() * 1000)

    def _wait_next_millis(self, last_timestamp: int) -> int:
        """
        自旋等待到下一毫秒，用于同毫秒序列号耗尽时的回退

        :param last_timestamp: 上次生成 ID 的时间戳
        :return: 不早于 last_timestamp + 1 的新时间戳
        """
        timestamp = self._current_millis()
        while timestamp <= last_timestamp:
            timestamp = self._current_millis()
        return timestamp

    def next_id(self) -> int:
        """
        生成并返回下一个雪花 ID

        同一毫秒内通过递增序列号保证唯一性；序列号耗尽则等待下一毫秒。
        检测到时钟回拨时，在容忍阈值内自旋等待时钟追平后继续，
        超出阈值抛出 RuntimeError 以阻止产生重复 ID。

        :return: 全局唯一、趋势递增的 64 位整型 ID
        """
        with self._lock:
            timestamp = self._current_millis()

            # 时钟回拨处理
            if timestamp < self._last_timestamp:
                backward = self._last_timestamp - timestamp
                if backward <= self.CLOCK_BACKWARD_TOLERANCE:
                    # 在容忍阈值内：自旋等待时钟追平
                    while timestamp < self._last_timestamp:
                        timestamp = self._current_millis()
                else:
                    raise RuntimeError(
                        f"时钟回拨 {backward}ms，超出容忍阈值 "
                        f"{self.CLOCK_BACKWARD_TOLERANCE}ms，拒绝生成 ID")

            # 同一毫秒内：序列号递增；跨毫秒：序列号归零
            if timestamp == self._last_timestamp:
                self._sequence = (self._sequence + 1) & self.SEQUENCE_MASK
                # 序列号回绕至 0，说明当前毫秒的 4096 个序号已耗尽，等待下一毫秒
                if self._sequence == 0:
                    timestamp = self._wait_next_millis(self._last_timestamp)
            else:
                self._sequence = 0

            self._last_timestamp = timestamp

            # 组装 64 位 ID：时间戳 | 数据中心ID | 机器ID | 序列号
            return (
                (timestamp - self.START_EPOCH) << self.TIMESTAMP_SHIFT
                | (self.data_center_id << self.DATA_CENTER_ID_SHIFT)
                | (self.worker_id << self.WORKER_ID_SHIFT)
                | self._sequence
            )

snow_flake = SnowFlake()


if __name__ == '__main__':
    # 模块级单例校验：重复导入得到同一实例
    from snow_flake import snow_flake as sf_a
    from snow_flake import snow_flake as sf_b
    print(f"模块单例校验: {sf_a is sf_b is snow_flake}")

    # 批量生成 ID，校验唯一性与趋势递增
    ids = [snow_flake.next_id() for _ in range(5)]
    print("生成示例:")
    for i, value in enumerate(ids):
        print(f"  next_id[{i}]: {value}")
    print(f"唯一性校验: {len(set(ids)) == len(ids)}")
    print(f"趋势递增校验: {ids == sorted(ids)}")

