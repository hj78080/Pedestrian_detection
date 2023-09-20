import redis

# Redis 클라이언트에서 모든 키 가져오기
r = redis.StrictRedis(host='localhost', port=6379, db=0)
keys = r.keys('*')

# key 값 중 최댓값 반환
def get_key_offset() :
    num = 0
    for key in keys:
        num = max(num, int(key))
    return num

# 모든 값 출력
def print_values() :
    for key in keys:
        value = r.get(key)
        print(f'Key: {key.decode("utf-8")}, Value: {value.decode("utf-8")}')

# 전체 삭제
def flush_DB() :
    r.flushdb()

print_values()