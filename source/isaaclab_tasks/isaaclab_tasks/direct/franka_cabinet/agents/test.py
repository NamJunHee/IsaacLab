from collections import deque

monster_num, total_turn = map(int, input().split())
grid_size = 4

packman_r, packman_c = map(int, input().split())
packman_r = packman_r - 1
packman_c = packman_c - 1

packman_id = 1;
monster_id = 2;
monster_egg_id = 3;
monster_dead_id = 4;

grid = [[ []for _ in range(grid_size)] for _ in range(grid_size)]
grid[packman_r][packman_c].append(packman_id)

input_monster_info = [list(map(int,input().split())) for _ in range(monster_num)]

# -1 해서 사용할 것
monster_info = deque()

for i in input_monster_info:
    monster_info.append(i)

for i in range(monster_num):
    monster_r = monster_info[i][0] - 1
    monster_c = monster_info[i][1] - 1
    grid[monster_r][monster_c].append(monster_id)

# -1 해서 사용할 것
monster_egg_info = deque()

# 1,2,3,4,5,6,7,8 
monster_dr = [0, -1, -1, 0, 1, 1, 1, 0, -1]
monster_dc = [0, 0, -1, -1, -1, 0, 1, 1, 1]

monster_dead_info = deque()

# 상 좌 하 우
packman_dr = [-1, 0, 1, 0]
packman_dc = [0, -1, 0, 1]

#function-----------------------------------------------------------------------------
def isOutofGrid(r,c):
    if r < 0 or r >= grid_size or c <0 or c >= grid_size:
        return True
    else:
        return False

def monster_copy_ready():
    for m_info in monster_info:
        monster_r = m_info[0] - 1
        monster_c = m_info[1] - 1
        monster_d = m_info[2]
        monster_egg_info.append([monster_r, monster_c, monster_d])

def monster_move():
    for i in range(monster_info):
        monster_r = monster_info[i][0] - 1
        monster_c = monster_info[i][1] - 1
        monster_d = monster_info[i][2]

        can_move = False

        new_mr = monster_r + monster_dr[monster_d]
        new_mc = monster_c + monster_dc[monster_d]

        for j in range(7):
            if isOutofGrid(new_mr, new_mc) or (packman_id in grid[new_mr][new_mc]) or (monster_dead_id in grid[new_mr][new_mc]):
                new_md = monster_d + 1
                if new_md >= 8:
                    new_md = 1
                else:
                    new_mr = monster_r + monster_dr[new_md]
                    new_mc = monster_c + monster_dc[new_md]
                    can_move = True
                    break

        if can_move:
            monster_info[i][0] = new_mr
            monster_info[i][1] = new_mc
            monster_info[i][2] = new_md
        else:
            continue

def packman_move(row, col):
    global packman_r, packman_c
    move_info = deque()
    score = 0

    move_direction = [0,1,2,3] # 상 좌 하 우
    for i in move_direction:
        for j in move_direction:
            for k in move_direction:

                first_r = row + packman_dr[i]
                first_c = col + packman_dc[i]

                if isOutofGrid(first_r, first_c):
                    continue
                if grid[first_r][first_c] == monster_id:
                    score += 1
                
                second_r = first_r + packman_dr[j]
                second_c = first_c + packman_dc[j]

                if isOutofGrid(second_r, second_c):
                    continue
                if grid[second_r][second_c] == monster_id:
                    score += 1

                third_r = second_r + packman_dr[j]
                third_c = second_c + packman_dc[j]

                if isOutofGrid(third_r, third_c):
                    continue
                if grid[third_r][third_c] == monster_id:
                    score += 1
                
                move_info.append([score,i,j,k])

    sort_move_info = sorted(move_info)
    packman_movement = sort_move_info[0]

    for i in range(3):
        packman_r = packman_r + packman_dr[packman_movement[i + 1]]
        packman_c = packman_c + packman_dc[packman_movement[i + 1]]

        if monster_id in grid[packman_r][packman_c]:
            grid[packman_r][packman_c].remove(monster_id)    

            remove_contition = [packman_r, packman_c]
            new_monster_info = deque(item for item in monster_info if item[:2] != remove_contition)
            monster_info = new_monster_info

            grid[packman_r][packman_c].append(monster_dead_id)
            monster_dead_info[packman_r][packman_c].append([packman_r,packman_c,2])
    
    for i in monster_dead_info:
        dead_r = monster_dead_info[i][0]
        dead_c = monster_dead_info[i][1]
        dead_time = monster_dead_info[i][2]

        dead_time -= 1
        if dead_time == 0:
            grid[dead_r][dead_c].remove(monster_dead_id)

def monster_copy_complete():
    for i in monster_egg_info:
        new_monster_r = i[0]
        new_monster_c = i[1]
        new_monster_d = i[2]

        monster_info.append([new_monster_r, new_monster_c, new_monster_d])

def cal_score():
    score = 0

    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] == monster_id:
                score += 1
    
    return score

#main----------------------------------------------------------------------------------
for now_turn in range(total_turn):
    monster_copy_ready()
    monster_move()
    packman_move(packman_r, packman_c)
    monster_copy_complete()

print(cal_score())
