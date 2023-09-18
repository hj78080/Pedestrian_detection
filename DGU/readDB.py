import redis

# Redis 클라이언트 생성
r = redis.StrictRedis(host='localhost', port=6379, db=0)

# 모든 키 가져오기
keys = r.keys('*')

# 각 키에 대해 값을 가져와 출력
for key in keys:
    value = r.get(key)
    print(f'Key: {key.decode("utf-8")}, Value: {value.decode("utf-8")}')