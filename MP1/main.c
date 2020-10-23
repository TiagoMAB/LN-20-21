#include <stdio.h>
#include <string.h>

int main() {

    FILE* f = fopen("TRAIN.txt", "r");
    FILE* labels = fopen("TRAIN-labels.txt", "w+");
    FILE* questions = fopen("TRAIN-questions.txt", "w+");
    char c;

    while (c != EOF) {
        while ((c = fgetc(f)) != ' ') {
            if (c == EOF) {
                break;
            }
            fprintf(labels, "%c", c);
        }
        
        fprintf(labels, "%c", '\n');
        while ((c = fgetc(f)) != '\n') {
            if (c == EOF) {
                break;
            }
            fprintf(questions, "%c", c);
        }  
    }

    fclose(f);
    fclose(labels);
    fclose(questions);

    return 0;
}