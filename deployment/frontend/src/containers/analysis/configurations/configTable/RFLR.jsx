import { Close } from "@mui/icons-material";
import {
    IconButton,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
} from "@mui/material";

const RFLR = ({ configurations, handleDelete }) => {
    return (
        <Table>
            <TableHead>
                <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Classification Threshold</TableCell>
                    <TableCell>Maximum Iterations</TableCell>
                    <TableCell>Maximum Time</TableCell>
                    <TableCell />
                </TableRow>
            </TableHead>
            <TableBody>
                {configurations.map((config, key) => (
                    <TableRow key={key}>
                        <TableCell>{config.name}</TableCell>
                        <TableCell>{config.threshold_classifier}</TableCell>
                        <TableCell>{config.max_iter}</TableCell>
                        <TableCell>{config.time_maximum}</TableCell>
                        <TableCell>
                            <IconButton
                                color="error"
                                onClick={() => handleDelete(key)}
                            >
                                <Close />
                            </IconButton>
                        </TableCell>
                    </TableRow>
                ))}
            </TableBody>
        </Table>
    );
};

export default RFLR;
