using System.Runtime.CompilerServices;
using PF1;

static void WRT<T>(T t) => Console.WriteLine(t); // wrapper to make Console.Writeline shorter to use
static void WRTA<T>(T[] a) =>
    a.ToList()
     .ForEach(t => { WRT(t); Console.WriteLine(); });


var A = new qMatrix<double>([
    [ -35,   0, -19,  99, -91, -35,   0],
    [   0,   0, -50,   0,   0,  80,  12],
    [   0, -88,   0,  -3,   0, -82,   0],
    [   0,   0,   0,   0,   0,  91,  22],
    [ -46,  18, -97,  95, -26,   0,   9],
    [   0,  86,   0, -61,  22,   0,   0],
    [   0,   0, -15, -78,   0,  82, -25]]);

var B = new qMatrix<double>(
    [
        [235],
        [24],
        [35],
        [1252],
        [1],
        [4],
        [1252],
    ]
);

WRT(M.MatRREF(M.MatRand(10, 10)));

WRTA(M.MatLu(A));

WRT(M.MatPow(A, 1));