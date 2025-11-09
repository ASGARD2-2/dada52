import random
import os


N = 25

stina="#"
pysto=" "
player="P"
exit="E"


maze=[[stina for _ in range(N)] for _ in range(N)]


def generate_maze():
    x,y=1,1
    maze[y][x]=pysto
    stack=[(x,y)]
    directions=[(-2,0),(2,0),(0,-2),(0,2)]
    while stack:
        x,y=stack[-1]
        random.shuffle(directions)
        for dx,dy in directions:
            nx,ny=x+dx,y+dy
            if 1<=nx<N-1 and 1<=ny<N-1 and maze[ny][nx]==stina:
                maze[y+dy//2][x+dx//2]=pysto
                maze[ny][nx]=pysto
                stack.append((nx,ny))
                break
        else:
            stack.pop()

def draw_maze(px,py):
    os.system('cls' if os.name=='nt' else 'clear')
    for y in range(N):
        for x in range(N):
            if x==px and y==py:
                print(player,end="")
            else:
                print(maze[y][x],end="")
        print()

generate_maze()
entrance=(1,1)
exit_pos=(N-2,N-2)
maze[exit_pos[1]][exit_pos[0]]=exit
player_x,player_y=entrance

while True:
    draw_maze(player_x,player_y)
    if (player_x,player_y)==exit_pos:
        print("ти знайшов вихід!!!")
        break
    move=input("рух(W/A/S/D): ").lower().strip()
    dx,dy=0,0
    if move=="w":dy=-1
    elif move=="s":dy=1
    elif move=="a":dx=-1
    elif move=="d":dx=1
    nx,ny=player_x+dx,player_y+dy
    if 0<=nx<N and 0<=ny<N and maze[ny][nx]!=stina:
        player_x,player_y=nx,ny