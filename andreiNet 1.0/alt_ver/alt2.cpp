
#include <graphics.h>

void drawnet()
{
    circle(100, 250, 50);
    circle(100, 450, 50);
    circle(100, 650, 50);


}



int main()
{
    initwindow(900, 900);
    drawnet();
    getch();
}
