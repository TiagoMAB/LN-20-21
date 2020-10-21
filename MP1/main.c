#include <stdio.h>
#include <string.h>

int main() {

    FILE* f = fopen("DEV.txt", "r");
    FILE* labels = fopen("DEV-labels.txt", "w+");
    FILE* questions = fopen("DEV-questions.txt", "w+");
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