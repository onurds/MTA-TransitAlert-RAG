# MTA GTFS Alerts Feed Documentation

**Last updated:** December 2021

This document defines the MTA's implementation of its GTFS-realtime alerts feed.

This document also contains extensions added specifically for the MTA and its content management system called Mercury (MercuryFeedHeader, MercuryAlert and MercuryEntitySelector). To use these extensions, you need the `mercury-gtfs-realtime.proto` file. Mercury had originally been called "Live Media Manager" so users may see legacy references to "LMM" in some fields.

Refer to the GTFS-realtime specification at <https://developers.google.com/transit/gtfs-realtime/> for more details on message field type, cardinality, etc. This document outlines only the Mercury specific usage and extensions.

This is the specification for version 1.0.

---

## *message* FeedMessage

The feed contains one kind of entity: Alerts

| Field Name | Usage |
|---|---|
| header | As defined in the GTFS-rt spec |
| entity | an array of alert messages |

### Examples

```json
"header": {
    "gtfsRealtimeVersion": "2.0",
    "incrementality": "FULL_DATASET",
    "timestamp": "1635792210",
    ".mercuryFeedHeader": {
        "mercuryVersion": "1.0"
    }
}
```

```json
"entity": [
    {
        "id": "lmm:alert:104907",
        "alert": {
            // alert info here
        }
    },
    {
        "id": "lmm:planned_work:1815",
        "alert": {
            // alert info here
        }
    }
]
```

---

## *message* FeedHeader

The feed is a full dataset and uses GTFS-realtime version 2.0

| Field Name | Usage |
|---|---|
| gtfs_realtime_version | 2.0 |
| incrementality | FULL_DATASET |
| timestamp | As defined in the GTFS-rt spec |

### Extensions

| Field Name | Type | Required | Cardinality | Description |
|---|---|---|---|---|
| mercury_feed_header | MercuryFeedHeader | Optional | One | Mercury-specific feed header information |

---

## *message* MercuryFeedHeader

This message is an extension to the FeedHeader containing the version of the Mercury extensions.

| Field Name | Type | Required | Cardinality | Description |
|---|---|---|---|---|
| mercury_version | string | Required | One | Version of the Mercury extensions specification. |

---

## *message* FeedEntity

As currently configured, the Feed Entity will be an alert. When an alert has multiple impacts, each impact will be displayed as a separate entity in the feed.

| Field Name | Usage |
|---|---|
| id | Format is `lmm:<id>` (e.g. `lmm:planned_work:1` or `lmm:alert_2`)* |
| is_deleted | Not currently used |
| trip_update | Not currently used |
| vehicle | Not currently used |
| alert | See Alert message |

> \* Note that `lmm:planned_work` indicates a planned service interruption, in the case of maintenance, repairs, upgrades, etc. and `lmm:alert` indicates an unplanned disruption of service.

---

## *message* Alert

An alert or planned work message indicating some sort of incident.

| Field Name | Usage |
|---|---|
| active_period | Array of times when the alert should be shown to the user, can be single or multiple ranges. If the alert does not have an end date set, the time range will only contain a start date. Times are formatted in Unix time. |
| informed_entity | See EntitySelector. |
| cause | Not currently used |
| effect | Not currently used |
| url | Not currently used |
| header_text | A headline to describe the high-level impacts of a disruption. Headlines are capped at 160 characters. |
| description_text | Provides full details of a disruption's impacts. Details do not have a character limit. |

Each headline is published with two translations:

- **"en"** headlines contain brackets around objects the MTA represents with symbology such as a route bullet or ADA icon. (e.g. "Delays on the \[F\] \[G\] trains.")
- **"en-html"** headlines contain both brackets for symbology and HTML to allow for basic formatting paragraphs, links, and other text styling.

`description_text` is published with two translations:

- **"en"** description_text contains plain text with brackets around objects the MTA represents with symbology such as a route bullet or ADA icon. (e.g. "Delays on the \[F\] \[G\] trains.")
- **"en-html"** description_text contains both brackets for symbology and HTML to allow for basic formatting paragraphs, links, and other text styling.

### Extensions

| Name | Type | Required | Cardinality | Description |
|---|---|---|---|---|
| transit_realtime.mercury_alert | MercuryAlert | Optional | One | Mercury-specific alert information |

### Example

```json
"alert": {
    "activePeriod": [
        {
            "start": "1637786700",
            "end": "1637788500"
        }
    ],
    "informedEntity": [
        {
            "agencyId": "LI",
            "routeId": "9",
            ".mercuryEntitySelector": {
                "sortOrder": "LI:9:10"
            }
        }
    ],
    "headerText": {
        "translation": [
            {
                "text": "Extra train for Thanksgiving Eve",
                "language": "en"
            },
            {
                "text": "Extra train for Thanksgiving Eve",
                "language": "en-html"
            }
        ]
    },
    "descriptionText": {
        "translation": [
            {
                "text": "// Description info here",
                "language": "en"
            },
            {
                "text": "// Description info here",
                "language": "en-html"
            }
        ]
    },
    ".mercuryAlert": {
        "createdAt": "1635548457",
        "updatedAt": "1635549994",
        "alertType": "Extra Service",
        "displayBeforeActive": "3600",
        "humanReadableActivePeriod": {
            "translation": [
                {
                    "text": "Wednesday, November 24",
                    "language": "en"
                }
            ]
        }
    }
}
```

---

## *message* transit_realtime.mercury_alert

This is an extension to the Alert that contains Mercury-specific information.

| Field Name | Type | Required | Cardinality | Description |
|---|---|---|---|---|
| created_at | uint64 | Required | One | Time when the message was created in Mercury (not to be confused with active_period start time). Times are formatted in Unix time. |
| updated_at | uint64 | Required | One | Time when the message was last updated in Mercury. Times are formatted in Unix time. |
| alert_type | string | Required | One | The service status category for the alert (e.g. "Delays"). While there are a standard set of service status categories, the MTA may add/remove/change them so data consumers should treat this as a free text field. |
| display_before_active | int | Required | | Number of seconds before the active_period start time that the MTA sets a message to appear on our homepage to give customers advance notice of a planned service change. The value for service alerts is 0 and the default value for planned work messages is 3600. |
| human_readable_active_period | A TranslatedString with one translation. Language specified as `en`. | Required for planned work messages | | A human-readable summary of the dates and times when a planned service change impacts customers. |
| additional_information | A TranslatedString with one translation. Language specified as `en`. | Deprecated | | Deprecated. Previously used in service alerts to display additional information about an alert. |
| station_alternative | An array of entities (affectedEntity objects with an agencyId and stopId) | | | An array of station alternatives for some planned work messages. Each station has an affectedEntity with agencyId and stopId and a notes object consisting of TranslatedStrings. |
| service_plan_number | Array | | | Internal service plans associated with the planned work message. |
| clone_id | String | | | If the message was duplicated from a previous message, this is the id of the original message. |

---

## *message* EntitySelector

A selector for an entity targeted in an alert.

| Name | Usage |
|---|---|
| agency_id | The GTFS-ID for the agency that correlated to the route_id or stop_id (e.g. MTASBWY). MTA agency IDs are MNR for Metro North Railroad, MTASBWY for subway, MTA NYCT or MTABC for buses, and LI for Long Island Rail Road. |
| route_id | Either route_id or stop_id will be set for each EntitySelector. Format is the GTFS-ID for the route without the agency prefix (e.g. G). |
| route_type | Not currently used |
| trip | Not currently used (coming soon for mentioned railroad trips) |
| stop_id | Either route_id or stop_id will be set for each EntitySelector. Format is the GTFS-ID for the route without the agency prefix (e.g. 10021). |

### Extensions

| Field Name | Type | Required | Cardinality | Description |
|---|---|---|---|---|
| mercury_entity_selector | MercuryEntitySelector | Optional | One | Mercury-specific entity selector information |

---

## *message* MercuryEntitySelector

This is an extension to the EntitySelector used for Mercury-specific entity information.

| Field Name | Type | Required | Cardinality | Description |
|---|---|---|---|---|
| sort_order | string | Required | One | Priority of the affected entity. Format is GTFS-ID:Priority (e.g. MTASBWY:G:16). Priority and NyctBusPriority enums are used to determine the priority number based on the alert type. Priority is in ascending order. |

---

## *enum* Priority

The relation between the alert type and the priority of the affected entity for NYCT Subway, MNR, and LIRR agencies.

---

## *enum* NyctBus Priority

The relation between the alert type and the priority of the affected entity for the NYCT Bus agency.
