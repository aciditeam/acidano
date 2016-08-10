/**
*
*  @file			needleman.c
*  @brief			Optimized Needleman-Wunsch alignment with homopolymers handling
*
*	This file contains the implementation of the optimized Needleman-Wunsch alignment algorithm with homopolymers handling
*
*  @author			Philippe Esling
*	@version		1.1
*	@date			16-01-2013
*  @copyright		UNIGE - GEN/EV (Pawlowski) - 2013
*	@licence		MIT Media Licence
*
*/

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#define U_FEPS 1.192e-6F          /* 1.0F + E_FEPS != 1.0F */
#define U_DEPS 2.22e-15           /* 1.0 +  E_DEPS != 1.0  */

#define E_FPEQ(a,b,e) (((b - e) < a) && (a < (b + e)))

#define _DIAG 	0
#define _UP  	1
#define _LEFT 	2

#define BUFFSIZE	4096
#define SEQSIZE		1024

#define PITCH_DIM     5
#define NUM_PITCH_CLASS     12

// Debug
#define LEN 10
#define BUF_SIZE 39

int		DNAFull[26][26] =
{{5,-4,-4,-1,0,0,-4,-1,0,0,-4,0,1,-2,0,0,0,1,-4,-4,-4,-1,1,0,-4,0},
{-4,-1,-1,-2,0,0,-1,-2,0,0,-1,0,-3,-1,0,0,0,-3,-1,-1,-1,-2,-3,0,-1,0},
{-4,-1,5,-4,0,0,-4,-1,0,0,-4,0,1,-2,0,0,0,-4,1,-4,-4,-1,-4,0,1,0},
{-1,-2,-4,-1,0,0,-1,-2,0,0,-1,0,-3,-1,0,0,0,-1,-3,-1,-1,-2,-1,0,-3,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{-4,-1,-4,-1,0,0,5,-4,0,0,1,0,-4,-2,0,0,0,1,1,-4,-4,-1,-4,0,-4,0},
{-1,-2,-1,-2,0,0,-4,-1,0,0,-3,0,-1,-1,0,0,0,-3,-3,-1,-1,-2,-1,0,-1,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{-4,-1,-4,-1,0,0,1,-3,0,0,-1,0,-4,-1,0,0,0,-2,-2,1,1,-3,-2,0,-2,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{1,-3,1,-3,0,0,-4,-1,0,0,-4,0,-1,-1,0,0,0,-2,-2,-4,-4,-1,-2,0,-2,0},
{-2,-1,-2,-1,0,0,-2,-1,0,0,-1,0,-1,-1,0,0,0,-1,-1,-2,-2,-1,-1,0,-1,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{1,-3,-4,-1,0,0,1,-3,0,0,-2,0,-2,-1,0,0,0,-1,-2,-4,-4,-1,-2,0,-4,0},
{-4,-1,1,-3,0,0,1,-3,0,0,-2,0,-2,-1,0,0,0,-2,-1,-4,-4,-1,-4,0,-2,0},
{-4,-1,-4,-1,0,0,-4,-1,0,0,1,0,-4,-2,0,0,0,-4,-4,5,5,-4,1,0,1,0},
{-4,-1,-4,-1,0,0,-4,-1,0,0,1,0,-4,-2,0,0,0,-4,-4,5,5,-4,1,0,1,0},
{-1,-2,-1,-2,0,0,-1,-2,0,0,-3,0,-1,-1,0,0,0,-1,-1,-4,-4,-1,-3,0,-3,0},
{1,-3,-4,-1,0,0,-4,-1,0,0,-2,0,-2,-1,0,0,0,-2,-4,1,1,-3,-1,0,-2,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{-4,-1,1,-3,0,0,-4,-1,0,0,-2,0,-2,-1,0,0,0,-4,-2,1,1,-3,-2,0,-1,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

int score_chord(long a, long b){
    int score = 0;
    char buffer[BUF_SIZE];
    buffer[BUF_SIZE - 1] = '\0';
    // score = number of bit on in the mask
    for (int i=0; i < NUM_PITCH_CLASS; i++)
    {
        score += (a & 0x1) & (b & 0x1);
        a = a >> 1;
        b = b >> 1;
    }
    return score;
}

static float getScore(const int *horizGap_m, const int *vertGap_m, const int *m, int lena, int lenb, int *start1, int *start2, int noEndGap_n)
{
    int i,j, cursor;
    float score = INT_MIN;
    *start1 = lena-1;
    *start2 = lenb-1;

    if(noEndGap_n)
    {
        cursor = lena * lenb - 1;
        if(m[cursor]>horizGap_m[cursor]&&m[cursor]>vertGap_m[cursor])
        score = m[cursor];
        else if(horizGap_m[cursor]>vertGap_m[cursor])
        score = horizGap_m[cursor];
        else
        score = vertGap_m[cursor];
    }
    else {

        for (i = 0; i < lenb; ++i)
        {
            cursor = (lena - 1) * lenb + i;
            if(m[cursor]>score)
            {
                *start2 = i;
                score = m[cursor];
            }
            if(horizGap_m[cursor]>score)
            {
                score = horizGap_m[cursor];
                *start2 = i;
            }
            if(vertGap_m[cursor]>score)
            {
                score = vertGap_m[cursor];
                *start2 = i;
            }
        }

        for (j = 0; j < lena; ++j)
        {
            cursor = j * lenb + lenb - 1;
            if(m[cursor]>score)
            {
                *start1 = j;
                *start2 = lenb-1;
                score = m[cursor];
            }
            if(horizGap_m[cursor]>score)
            {
                score = horizGap_m[cursor];
                *start1 = j;
                *start2 = lenb-1;
            }
            if(vertGap_m[cursor]>score)
            {
                score = vertGap_m[cursor];
                *start1 = j;
                *start2 = lenb-1;
            }
        }
    }
    return score;
}

int needlemanWunsch(const long *a, const long *b, int lena, int lenb, long *trace_a, long *trace_b, int gapopen, int gapextend)
{
    // Tuning parameters
    int     noEndGap_n = 1;
    int     endgapopen = 0;
    int     endgapextend = 0;

    int     curMalloc = lena * lenb;
    int     *m = malloc(curMalloc * sizeof(int));
    int     *horizGap_m = malloc(curMalloc * sizeof(int));
    int     *vertGap_m = malloc(curMalloc * sizeof(int));
    int     *trBack = malloc(curMalloc * sizeof(int));

    int		xpos, ypos;
    int     counter;
    long    bconvcode;
    int		match;
    int		horizGap_mp;
    int		vertGap_mp;
    int		mp;
    int		cursor = 0, cursorp;
    int		*start1, *start2;
    int		testog;
    int		testeg;
    int     bestSoFar;

    // printf("Gapopen = %i\n", gapopen);
    // printf("Gapextend = %i\n", gapextend);

    if (noEndGap_n == 1)
    {
        endgapopen = 0;
        endgapextend = 0;
    }
    start1 = calloc(1, sizeof(int));
    start2 = calloc(1, sizeof(int));
    horizGap_m[0] = -endgapopen-gapopen;
    vertGap_m[0] = -endgapopen-gapopen;


    // m[0] = DNAFull[a[0] - 'A'][b[0] - 'A'];
    m[0] = score_chord(a[0],b[0]);

    /* First initialise the first column */
    for (ypos = 1; ypos < lena; ++ypos)
    {
        // match = DNAFull[a[ypos] - 'A'][b[0] - 'A'];
        match = score_chord(a[ypos],b[0]);
        cursor = ypos * lenb;
        cursorp = cursor - lenb;
        testog = m[cursorp] - gapopen;
        testeg = vertGap_m[cursorp] - gapextend;
        vertGap_m[cursor] = (testog >= testeg ? testog : testeg);
        m[cursor] = match - (endgapopen + (ypos - 1) * endgapextend);
        horizGap_m[cursor] = - endgapopen - ypos * endgapextend - gapopen;
    }
    horizGap_m[cursor] -= endgapopen - gapopen;
    // Then first row
    for (xpos = 1; xpos < lenb; ++xpos)
    {
        // match = DNAFull[a[0] - 'A'][b[xpos] - 'A'];
        match = score_chord(a[0],b[xpos]);
        cursor = xpos;
        cursorp = xpos-1;
        testog = m[cursorp] - gapopen;
        testeg = horizGap_m[cursorp] - gapextend;
        horizGap_m[cursor] = (testog >= testeg ? testog : testeg);
        m[cursor] = match - (endgapopen + (xpos - 1) * endgapextend);
        vertGap_m[cursor] = -endgapopen - xpos * endgapextend -gapopen;
    }
    vertGap_m[cursor] -= endgapopen - gapopen;
    xpos = 1;
    /*
    * Filling step
    */
    while (xpos != lenb)
    {
        ypos = 1;
        bconvcode = b[xpos];
        cursorp = xpos-1;
        cursor = xpos++;
        bestSoFar = INT_MAX;
        while (ypos < lena)
        {
            match = score_chord(a[ypos++],bconvcode);
            cursor += lenb;
            mp = m[cursorp];
            horizGap_mp = horizGap_m[cursorp];
            vertGap_mp = vertGap_m[cursorp];
            if(mp > horizGap_mp && mp > vertGap_mp)
            m[cursor] = mp+match;
            else if(horizGap_mp > vertGap_mp)
            m[cursor] = horizGap_mp+match;
            else
            m[cursor] = vertGap_mp+match;
            if(xpos==lenb)
            {
                testog = m[++cursorp] - endgapopen;
                testeg = vertGap_m[cursorp] - endgapextend;
            }
            else
            {
                testog = m[++cursorp];
                if (testog<horizGap_m[cursorp])
                testog = horizGap_m[cursorp];
                testog -= gapopen;
                testeg = vertGap_m[cursorp] - gapextend;
            }
            if(testog > testeg)
            vertGap_m[cursor] = testog;
            else
            vertGap_m[cursor] = testeg;
            cursorp += lenb;
            if(ypos==lena)
            {
                testog = m[--cursorp] - endgapopen;
                testeg = horizGap_m[cursorp] - endgapextend;
            }
            else
            {
                testog = m[--cursorp];
                if (testog<vertGap_m[cursorp])
                testog = vertGap_m[cursorp];
                testog -= gapopen;
                testeg = horizGap_m[cursorp] - gapextend;
            }
            if (testog > testeg)
            horizGap_m[cursor] = testog;
            else
            horizGap_m[cursor] = testeg;
        }
    }
    getScore(horizGap_m, vertGap_m, m, lena, lenb, start1, start2, noEndGap_n);
    xpos = *start2;
    ypos = *start1;
    cursorp = 0;

    // Prompt
    // for(int i=0; i<lena*lenb;i++)
    // {
    //     printf("%i; ", m[i]);
    //     if(i % (lena+1) == lena)
    //     {
    //         printf("\n");
    //     }
    // }
    /*
    * Trace-back step
    */
    // Keep track of the indices
    counter = 0;
    while (xpos>=0 && ypos>=0)
    {
        // Store index evolution
        // trace_a[lenb + lena - counter] = (long) ypos; // a take y... don't know the fuck why
        // trace_b[lenb + lena - counter] = (long) xpos;
        //
        // counter++;

        cursor = ypos*lenb+xpos;
        mp = m[cursor];
        if(cursorp == _LEFT && E_FPEQ((ypos==0||(ypos==lena)?
        endgapextend:gapextend), (horizGap_m[cursor]-horizGap_m[cursor+1]),U_FEPS))
        {
            trBack[cursor] = _LEFT;
            xpos--;
        }
        else if(cursorp== _UP && E_FPEQ((xpos==0||(xpos==lenb)?
        endgapextend:gapextend), (vertGap_m[cursor]-vertGap_m[cursor+lenb]),U_FEPS))
        {
            trBack[cursor] = _UP;
            ypos--;
        }
        else if(mp >= horizGap_m[cursor] && mp >= vertGap_m[cursor])
        {
            if(cursorp == _LEFT && E_FPEQ(mp,horizGap_m[cursor],U_FEPS))
            {
                trBack[cursor] = _LEFT;
                xpos--;
            }
            else if(cursorp == _UP && E_FPEQ(mp,vertGap_m[cursor],U_FEPS))
            {
                trBack[cursor] = _UP;
                ypos--;
            }
            else
            {
                trBack[cursor] = _DIAG;
                ypos--;
                xpos--;
            }
        }
        else if(horizGap_m[cursor]>=vertGap_m[cursor] && xpos>-1)
        {
            trBack[cursor] = _LEFT;
            xpos--;
        }
        else if(ypos>-1)
        {
            trBack[cursor] = _UP;
            ypos--;
        }
        cursorp = trBack[cursor];

        // Write and gap alignment along the sequence
        if(trBack[cursor] == _UP)
        {
            // Gap in the second sequence
            trace_a[lenb + lena - counter] = 1;
            trace_b[lenb + lena - counter] = 0;
        }
        else if(trBack[cursor] == _LEFT)
        {
            // Gap in the second sequence
            trace_a[lenb + lena - counter] = 0;
            trace_b[lenb + lena - counter] = 1;
        }
        else if(trBack[cursor] == _DIAG)
        {
            // Gap in the second sequence
            trace_a[lenb + lena - counter] = 1;
            trace_b[lenb + lena - counter] = 1;
        }
        counter++;

    }

    return counter;
}

/* ################################################ */
/* Python interface */
static PyObject	*needleman_chord(PyObject* self, PyObject* args)
{

    // Debug
    FILE *f = fopen("file2.txt", "w"); fclose(f);

    // Input
    PyObject    *alist_PY; /* the list of int */
    PyObject    *blist_PY;
    int         gapopen;
    int         gapextend;
    // One element in a or b (needed for parsing a and b)
    PyObject    *intObj;
    // Python list converted to long list
    long        *alist;
    long        *blist;
    // Lists lenght
    int         lena;
    int         lenb;

    // Trace length
    int trace_length;
    // Trace indexes for warping the tracks
    long        *trace_a;
    long        *trace_b;
    // Converted in Python object
    PyObject    *trace_a_PY; /* the list of int */
    PyObject    *trace_b_PY;

    // Get input value from python script
    if (!PyArg_ParseTuple(args, "OOii", &alist_PY, &blist_PY, &gapopen, &gapextend))
    {
        printf("Bad format for argument\n");
        return NULL;
    }

    /* get the number of lines passed to us */
    lena = (int) PyList_Size(alist_PY);
    if (lena < 0)   return NULL; /* Not a list */
    lenb = (int) PyList_Size(blist_PY);
    if (lenb < 0)   return NULL; /* Not a list */

    // Initialize alist and blist
    alist = calloc(lena, sizeof(long));
    blist = calloc(lenb, sizeof(long));
    // Trace_a and trace_b
    trace_a = calloc(lena+lenb, sizeof(long));
    trace_b = calloc(lena+lenb, sizeof(long));

    // printf("Size of a : %i\n", lena);
    // printf("Size of b : %i\n", lenb);

    // Build int lists
    /* iterate over items of the list, grabbing strings, and parsing
    for numbers */
    for (int i=0; i<lena; i++){
        /* grab the string object from the next element of the list */
        intObj = PyList_GetItem(alist_PY, i); /* Can't fail */
        /* make it a string */
        alist[i] = PyInt_AsLong(intObj);
        /* now do the parsing */
    }

    for (int i=0; i<lenb; i++){
        /* grab the string object from the next element of the list */
        intObj = PyList_GetItem(blist_PY, i); /* Can't fail */
        /* make it a string */
        blist[i] = PyInt_AsLong(intObj);
        /* now do the parsing */
    }

    // Call function
    trace_length = needlemanWunsch(alist, blist, lena, lenb, trace_a, trace_b, gapopen, gapextend);

    // printf("Trace length : %i\n", trace_length);

    trace_a_PY = PyList_New(trace_length);
    if (!trace_a_PY)
    return NULL;
    for (int i = 0; i < trace_length; i++) {
        // Index are lenb+lena-i to flip lr the list and return only useful part
        PyObject *intObj = PyInt_FromLong(trace_a[lenb + lena - i]);
        if (!intObj) {
            Py_DECREF(trace_a_PY);
            return NULL;
        }
        // Be carreful with order : new element is added at (trace_length-i)
        PyList_SET_ITEM(trace_a_PY, (trace_length-1-i), intObj);
    }

    trace_b_PY = PyList_New(trace_length);
    if (!trace_b_PY)
    return NULL;
    for (int i = 0; i < trace_length; i++) {
        PyObject *intObj = PyInt_FromLong(trace_b[lenb + lena - i]);
        if (!intObj) {
            Py_DECREF(trace_b_PY);
            return NULL;
        }
        PyList_SET_ITEM(trace_b_PY, (trace_length-1-i), intObj);
    }

    // Convert to python value output
    return Py_BuildValue("OO", trace_a_PY, trace_b_PY);
}

static PyMethodDef NeedleMethods[] = {
    {"needleman_chord", (PyCFunction)needleman_chord, METH_VARARGS, "Calculate Needleman-Wunsch for a whole set"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC  // Precise return type = void + declares any special linkage declarations required by the platform
initneedleman_chord(void)
{
    PyObject *module;
    module = Py_InitModule("needleman_chord", NeedleMethods);

    if (module == NULL)
    return;
}

/* ################################################ */
/* DEBUG */
// int main(){
//     FILE *f = fopen("file2.txt", "w"); fclose(f);
//     int res;
//     res = score_chord((long)109,(long)103);
//     fprintf(f, "%i\n", res);
//     return 0;
// }
