random_pitch() {
    local delta=${1:-50}
    local value=$(( ($RANDOM % $delta) - $delta/2 ))
    echo "+${value}" | sed 's/+-/-/'
}

glados() {
    local pitch=80
    local speed=170
    local lang=en
    local voice=f5
    local output=
    local text="Hello, and, again, welcome to the Aperture Science Computer-aided Enrichment Center"
    while true; do
        case "$1" in
            -v | --voice )
                voice="$2"
                shift 2
                ;;
            -l | --language )
                lang="$2"
                shift 2;
                ;;
            -p | --pitch )
                pitch="$2"
                shift 2;
                ;;
            -s | --speed )
                speed="$2"
                shift 2
                ;;
            -o | --output)
                output="-w $2"
                shift 2
                ;;
            -- ) 
                shift
                break
                ;;
            * ) 
                if [ -z "$1" ]; then
                    break;
                fi
                text="$1"
                shift
                ;;
        esac
    done

    local word_pitch=0
    local prosody_data=""
    for word in $text; do
        word_pitch=$(random_pitch 60)
        prosody_data="${prosody_data}<prosody pitch=\"${word_pitch}\">${word}</prosody>";
    done 

    espeak "${prosody_data}" -m -p ${pitch} -s ${speed} -v "${lang}+${voice}" ${output}
}

glados "$@"
