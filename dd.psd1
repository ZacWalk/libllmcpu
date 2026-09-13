@{
    schema = 1
    project = @{
        name = 'nnc'
        type = 'cli'
        'default-target' = 'nnc'
    }
    dependencies = @{ owner = 'dd' }
    build = @{
        'x64-windows' = @{
            debug = 'debug'
            release = 'release'
        }
        'x64-linux' = @{
            debug = 'debug'
            release = 'release'
        }
    }
    targets = @(
        @{
            id = 'nnc'
            kind = 'cli'
            'cmake-target' = 'nnc'
            'test-label' = 'nnc'
            'debug-path' = 'exe/nnc-d{exe}'
            'release-path' = 'exe/nnc{exe}'
        }
    )
    commands = @{
        download = @{
            description = 'Download a GGUF weight file into ./models. No --name lists the known models. Network unless already present.'
            script = 'cmake/project-command.ps1'
            effects = 'write'
            'supports-dry-run' = $true
            'timeout-secs' = 7200
            parameters = @{
                name = @{ type = 'string'; description = "Model key, or 'all'. Omit to list the catalogue." }
            }
        }
    }
}
