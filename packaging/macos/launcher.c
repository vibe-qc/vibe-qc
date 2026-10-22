/* Relocatable command launcher. SPDX-License-Identifier: MPL-2.0 */
#include <mach-o/dyld.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern char **environ;

int main(int argc, char **argv) {
    char path[PATH_MAX], resolved[PATH_MAX], resources[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) || !realpath(path, resolved)) {
        perror("Cannot locate the application"); return 1;
    }
    if (snprintf(resources, sizeof(resources), "%s/../Resources", dirname(resolved))
            >= (int)sizeof(resources)) return 1;
    /* Remove interpreter and loader overrides before exec, including child runs. */
    for (int i = 0; environ[i];) {
        if (!strncmp(environ[i], "PYTHON", 6) || !strncmp(environ[i], "DYLD_", 5)
                || !strncmp(environ[i], "VIRTUAL_ENV=", 12)
                || !strncmp(environ[i], "__PYVENV_LAUNCHER__=", 19)
                || !strncmp(environ[i], "RDBASE=", 7)) {
            char *key = strdup(environ[i]);
            if (!key) return 1;
            *strchr(key, '=') = '\0'; unsetenv(key); free(key);
        } else i++;
    }
    char command[PATH_MAX];
    if (!strcmp(basename(argv[0]), "Launcher")) {
        snprintf(command, sizeof(command), "%s/Start.command", resources);
        execl("/usr/bin/open", "open", "-a", "Terminal", command, (char *)NULL);
    } else {
        char python[PATH_MAX];
        snprintf(python, sizeof(python), "%s/python/bin/python3", resources);
        snprintf(command, sizeof(command), "%s/run_entrypoint.py", resources);
        char **args = calloc((size_t)argc + 6, sizeof(char *));
        if (!args) return 1;
        args[0] = python; args[1] = "-I"; args[2] = "-B";
        args[3] = command; args[4] = basename(argv[0]);
        for (int i = 1; i < argc; i++) args[i + 4] = argv[i];
        execv(python, args);
    }
    perror("Unable to launch bundled runtime; reinstall the application");
    return 1;
}
