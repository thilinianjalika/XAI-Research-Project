import commonStyles from "@/styles/commonStyles";
import { Box, IconButton, TextField, Typography } from "@mui/material";
import Textarea from "@/components/Textarea";
import { useContext, useEffect, useRef, useState } from "react";
import { startTestCase } from "@/functions/api";
import { LoadingButton } from "@mui/lab";
import Configurations from "./configurations/Configurations";
import { ModalContext } from "@/providers/modalProvider/ModalProvider";
import { MODEL_NAME_KNN, MODEL_NAME_SVM } from "@/constants";
import { Close } from "@mui/icons-material";
import styles from "./styles";

const Analysis = ({ model }) => {
    const { setNotification } = useContext(ModalContext);
    const textareaRef = useRef();
    const variationsRef = useRef();
    const [loading, setLoading] = useState(false);
    const [configurations, setConfigurations] = useState([]);
    const [report, setReport] = useState();

    const evaluationHandler = () => {
        const prompt = textareaRef.current && textareaRef.current.value;
        const variations = variationsRef.current
            ? parseInt(variationsRef.current.value)
            : null;
        if (prompt !== "" && model && variations !== 0) {
            setLoading(true);
            startTestCase({
                model_name: model,
                prompt,
                variations,
                configurations,
            })
                .then(setReport)
                .catch(setNotification)
                .finally(() => {
                    setLoading(false);
                });
        }
    };
    useEffect(() => {
        setConfigurations([]);
        setReport();
    }, [model]);
    return (
        <Box sx={commonStyles.sectionContainer}>
            <Typography variant="h3">Analysis</Typography>
            <Textarea
                ref={textareaRef}
                placeholder="Prompt"
                sx={commonStyles.text}
            />
            {(model === MODEL_NAME_SVM || model === MODEL_NAME_KNN) && (
                <TextField
                    inputRef={variationsRef}
                    sx={commonStyles.text}
                    label="Variations"
                    size="small"
                    type="number"
                    defaultValue={2}
                />
            )}
            <Typography variant="h4">Test Cases</Typography>
            <Configurations
                model={model}
                configurations={configurations}
                setConfigurations={setConfigurations}
            />
            <LoadingButton
                loading={loading}
                sx={commonStyles.btn}
                onClick={evaluationHandler}
            >
                Analyze
            </LoadingButton>
            {report && (
                <>
                    <Typography variant="h4">Report</Typography>
                    <Box sx={commonStyles.outputContainer()}>
                        <IconButton
                            sx={styles.close}
                            onClick={() => setReport()}
                        >
                            <Close />
                        </IconButton>
                        <Typography variant="body1" sx={commonStyles.codeBlock}>
                            {report}
                        </Typography>
                    </Box>
                </>
            )}
        </Box>
    );
};

export default Analysis;
